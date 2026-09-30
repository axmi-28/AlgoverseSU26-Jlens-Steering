"""C22: isolate target writing, swap-coordinate changes, and edit magnitude.

All controls use the same clean activations. This module never modifies a
cached lens or the historical swap implementation. See notes/c22-design.md.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from jlens.hooks import ActivationRecorder

from jsteer.config import load_config
from jsteer.data import probe_swap_items
from jsteer.loading import first_token_id, load_model, unembedding_rows
from jsteer.steering import ResidualEdit, answer_matches, greedy_continuation

ARMS = (
    "swap_full",
    "swap_off",
    "swap_off_matched",
    "swap_projected_matched",
    "swap_random_projection_matched",
    "frozen_full",
    "frozen_off",
    "frozen_logit",
    "frozen_random",
    "add_full",
    "add_off",
)


def unit(v: torch.Tensor) -> torch.Tensor:
    n = v.norm()
    if not torch.isfinite(v).all() or n < 1e-10:
        raise ValueError("Nonfinite or degenerate direction; cannot normalize")
    return v / n


def remove_projection(q: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    z = unit(axis)
    return q - (q @ z) * z


def random_projection_axis(q, axis, generator):
    """Random orientation removing the SAME norm as projection onto axis."""
    qhat = unit(q)
    cosine = (qhat @ unit(axis)).abs().clamp(0, 1)
    z = torch.randn(q.shape, generator=generator, dtype=q.dtype)
    orthogonal = unit(remove_projection(z, qhat))
    return cosine * qhat + torch.sqrt((1 - cosine.square()).clamp_min(0)) * orthogonal


def coordinates(h, v, ridge=1e-6):
    return torch.linalg.solve(
        v.T @ v + ridge * torch.eye(2, dtype=v.dtype), (h @ v).T
    ).T


def swap_delta(h, v, ridge=1e-6):
    c = coordinates(h, v, ridge)
    return (c.flip(-1) - c) @ v.T


def match_row_norms(delta, reference):
    norm = delta.norm(dim=-1, keepdim=True)
    budget = reference.norm(dim=-1, keepdim=True)
    if ((norm < 1e-10) & (budget > 1e-8)).any():
        raise ValueError("A zero edit cannot be matched to a nonzero budget")
    return delta * (budget / norm.clamp_min(1e-10))


def layer_controls(h, full, off, unembedding, *, seed=0):
    """Return [position, d] edits; full/off/unembedding have shape [d, 2].

    frozen_* preserve full-lens read coefficients and its source term, and
    match the target column's norm before substituting a write direction.
    Their TOTAL edit norms need not match: cancellation with the source term
    is part of the effect. add_* are positive, target-only, per-row norm-matched.
    """
    h, full, off, unembedding = [
        x.detach().cpu().float() for x in (h, full, off, unembedding)
    ]
    if h.ndim != 2 or any(x.shape != (h.shape[1], 2) for x in (full, off, unembedding)):
        raise ValueError("Expected h=[positions,d], direction pairs=[d,2]")
    generator = torch.Generator().manual_seed(seed)
    ref = swap_delta(h, full)
    off_delta = swap_delta(h, off)
    projected = torch.stack(
        [remove_projection(full[:, j], unembedding[:, j]) for j in range(2)], dim=1
    )
    random_projected = torch.stack(
        [
            remove_projection(
                full[:, j],
                random_projection_axis(full[:, j], unembedding[:, j], generator),
            )
            for j in range(2)
        ],
        dim=1,
    )
    c = coordinates(h, full)
    d = c[:, 0] - c[:, 1]
    source = -d[:, None] * full[:, 0]
    target_norm = full[:, 1].norm()
    writers = {
        "full": full[:, 1],
        "off": target_norm * unit(off[:, 1]),
        "logit": target_norm * unit(unembedding[:, 1]),
        "random": target_norm
        * unit(torch.randn(full[:, 1].shape, generator=generator)),
    }
    edits = {
        "swap_full": ref,
        "swap_off": off_delta,
        "swap_off_matched": match_row_norms(off_delta, ref),
        "swap_projected_matched": match_row_norms(swap_delta(h, projected), ref),
        "swap_random_projection_matched": match_row_norms(
            swap_delta(h, random_projected), ref
        ),
        **{f"frozen_{name}": source + d[:, None] * v for name, v in writers.items()},
        "add_full": ref.norm(dim=-1, keepdim=True) * unit(full[:, 1]),
        "add_off": ref.norm(dim=-1, keepdim=True) * unit(off[:, 1]),
    }
    for value in edits.values():
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite control edit")
    return edits


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes():
    source_dir = Path(__file__).parent
    return {
        name: file_hash(source_dir / name)
        for name in (
            "write_controls.py",
            "steering.py",
            "loading.py",
            "data.py",
            "config.py",
        )
    }


@dataclass
class WriteControlSpec:
    directions_path: str
    reference_path: str
    config_name: str = "qwen3-8b"
    corpus: str = "gsm8k"
    strengths: tuple[float, ...] = (0.5, 1.0)
    layers: tuple[int, ...] = (13, 16, 19, 22, 25, 28, 31)
    limit: int = 8
    seed: int = 2201
    gen_tokens: int = 6
    max_seconds: int = 1800
    out_dir: str = "results/c22"
    expected_sources: dict[str, str] | None = None


def run_write_controls(spec: WriteControlSpec) -> dict:
    """Resume only exact-manifest runs; checkpoint complete items atomically."""
    if (
        spec.limit < 1
        or not spec.strengths
        or any(not math.isfinite(s) or s <= 0 for s in spec.strengths)
    ):
        raise ValueError("Use a positive item limit and finite positive strengths")
    if spec.max_seconds < 1 or spec.gen_tokens < 1:
        raise ValueError("Positive runtime and generation limits required")
    if len(set(spec.strengths)) != len(spec.strengths) or len(set(spec.layers)) != len(
        spec.layers
    ):
        raise ValueError("Duplicate strengths or layers")
    if spec.expected_sources is not None and spec.expected_sources != source_hashes():
        raise ValueError("Deployed source differs from launcher's source; redeploy")
    config = load_config(spec.config_name)
    blob = torch.load(spec.directions_path, map_location="cpu", weights_only=False)
    targets = blob["targets"]
    if len(set(targets)) != len(targets):
        raise ValueError("Duplicate target words in component artifact")
    index = {w: i for i, w in enumerate(targets)}
    matrices = {}
    counts = {}
    for part in ("full", "off"):
        for layer in spec.layers:
            entry = blob["digests"][f"{spec.corpus}_{part}|{layer}"]
            matrix = entry["pullbacks"].float()
            if matrix.shape != (len(targets), config.d_model):
                raise ValueError(
                    "Component dimensions do not match model and target list"
                )
            matrices[part, layer] = matrix
            counts[f"{part}|{layer}"] = entry["n"]
    if len(set(counts.values())) != 1:
        raise ValueError("Full/off artifact fit counts differ")
    reference = json.loads(Path(spec.reference_path).read_text())
    clean_names = {
        r["name"] for r in reference if r["arm"] == "baseline" and r["baseline_ok"]
    }
    available = [
        item
        for item in probe_swap_items()
        if item.name in clean_names
        and item.intermediate in index
        and item.swap_to in index
    ]
    # Hash ordering is independent of observed steered outcomes.
    available.sort(
        key=lambda item: hashlib.sha256(f"{spec.seed}|{item.name}".encode()).hexdigest()
    )
    selected = available[: spec.limit]
    if not selected:
        raise ValueError("No items with clean baseline and component coverage")
    manifest = {
        "spec": asdict(spec),
        "config": asdict(config),
        "arms": ARMS,
        "items": [asdict(item) for item in selected],
        "eligible_items": len(available),
        "fit_counts": counts,
        "directions_sha256": file_hash(spec.directions_path),
        "reference_sha256": file_hash(spec.reference_path),
        "source_sha256": source_hashes(),
    }
    manifest = json.loads(json.dumps(manifest))
    tag = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    out = Path(spec.out_dir) / f"write_controls_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"manifest": manifest, "records": [], "status": "running"}
    if out.exists():
        payload = json.loads(out.read_text())
        if payload["manifest"] != manifest:
            raise ValueError("Existing output does not match the run manifest")

    def checkpoint():
        temporary = out.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
        temporary.replace(out)

    expected = 1 + len(ARMS) * len(spec.strengths)
    expected_keys = {("baseline", 0.0)} | {
        (a, float(s)) for a in ARMS for s in spec.strengths
    }
    done = set()
    for item in selected:
        rows = [r for r in payload["records"] if r["name"] == item.name]
        keys = {(r["arm"], r["strength"]) for r in rows}
        if rows and (len(rows) != expected or keys != expected_keys):
            raise ValueError(f"Incomplete or duplicate saved item: {item.name}")
        if rows:
            done.add(item.name)
    start = time.monotonic()
    if len(done) != len(selected):
        model = load_model(config)
        for item in selected:
            if item.name in done:
                continue
            if time.monotonic() - start > spec.max_seconds:
                break
            ids = [first_token_id(model, w) for w in (item.intermediate, item.swap_to)]
            u = unembedding_rows(model, ids).T.cpu().float()
            input_ids = model.encode(item.prompt, max_length=config.fit.max_seq_len)
            with (
                torch.no_grad(),
                ActivationRecorder(model.layers, at=list(spec.layers)) as rec,
            ):
                model.forward(input_ids)
                h = {
                    l: rec.activations[l][0].detach().cpu().float() for l in spec.layers
                }
            banks = {}
            si, ti = index[item.intermediate], index[item.swap_to]
            for layer in spec.layers:
                banks[layer] = layer_controls(
                    h[layer],
                    matrices["full", layer][[si, ti]].T,
                    matrices["off", layer][[si, ti]].T,
                    u,
                    seed=spec.seed + layer,
                )
            base = asdict(item)
            base["token_ids"] = ids
            base["token_counts"] = [
                len(model.tokenizer.encode(" " + w, add_special_tokens=False))
                for w in (item.intermediate, item.swap_to)
            ]
            rows = []
            settings = [("baseline", 0.0)] + [
                (arm, float(s)) for arm in ARMS for s in spec.strengths
            ]
            for arm, strength in settings:
                # Enforce the time budget inside an item as well; incomplete
                # item results are not committed, so a resumed run stays paired.
                if time.monotonic() - start > spec.max_seconds:
                    break
                edit = (
                    None
                    if arm == "baseline"
                    else ResidualEdit(
                        {l: banks[l][arm] * strength for l in spec.layers},
                        positions=list(range(input_ids.shape[1])),
                    )
                )
                text = greedy_continuation(
                    model,
                    item.prompt,
                    edit=edit,
                    n_tokens=spec.gen_tokens,
                    max_seq_len=config.fit.max_seq_len,
                )
                row = {
                    **base,
                    "arm": arm,
                    "strength": strength,
                    "generated": text,
                    "hit_answer": answer_matches(text, item.answer),
                    "hit_swap_answer": answer_matches(text, item.swap_answer),
                    "emitted_substring": item.swap_to.casefold() in text.casefold(),
                    "edit_row_norms": {}
                    if edit is None
                    else {
                        str(l): v.norm(dim=-1).tolist() for l, v in edit.vectors.items()
                    },
                }
                rows.append(row)
            if len(rows) != expected:
                break
            payload["records"].extend(rows)
            done.add(item.name)
            checkpoint()
            print(
                f"C22 {len(done)}/{len(selected)} items complete, {time.monotonic() - start:.1f}s",
                flush=True,
            )
    payload["status"] = "complete" if len(done) == len(selected) else "time_limit"
    payload["last_attempt_seconds"] = round(time.monotonic() - start, 2)
    checkpoint()
    return {
        "path": str(out),
        "status": payload["status"],
        "items_done": len(done),
        "items_selected": len(selected),
        "records": len(payload["records"]),
    }
