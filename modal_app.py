"""Modal entrypoints for the three GPU sweeps.

This file is deployment only. Every line of science lives in ``jsteer.sweeps``
and is called unchanged from here, so the code that costs money is the code the
local test suite exercises.

Why the pieces are shaped the way they are
------------------------------------------
**Two volumes, not one.** ``jsteer-cache`` holds the 16 GB of Qwen3-8B weights,
the 1.2 GB lens, and the materialized wikitext control list. Without it every
container re-downloads all of that, which on an 8-way fan-out costs more in
cold-start GPU-seconds than the sweep itself. ``jsteer-results`` holds the
output and is what you pull down at the end. Keeping them separate means you
can wipe results and re-run without re-warming 17 GB.

**The corpus list is materialized into the cache volume, not streamed per
container.** Each shard builds the full prompt list and strides it, so all
shards must agree on which wikitext document is prompt 5. Reading one file
makes that structural instead of a bet on the HuggingFace stream being
deterministic across eight machines.

**Sharding is a stride, not a block.** See ``jsteer.sweeps.take_shard``: the
prompt set mixes 8-token eval prompts with 128-token wikitext controls, and a
contiguous slice would hand one container every expensive prompt.

**Per-second billing makes fan-out nearly free.** Eight shards cost the same
GPU-seconds as one and finish eight times sooner. The wall-time win is real;
the dollar cost is roughly flat.

**One deployed app, never ``modal run``.** ``modal run`` creates a new
*ephemeral* app for every invocation, each of which becomes its own permanent
dashboard entry -- roughly sixty accumulated under ``jsteer`` before it was
noticed, and Modal exposes no API to delete an app record (see :func:`_fn`).
So the app is deployed once and every launch resolves its functions by name,
which creates function invocations under the single existing app and no new
app. Redeploy whenever the image or a remote function body changes; changing a
*launcher* needs no redeploy, since launchers run locally.

**`modal deploy` completing is NOT evidence that your source shipped.** It will
happily reuse a cached image and print "App deployed" in the same ~20 s it
takes when it rebuilds. That happened here: a fixed grading rule did not reach
the container, the run completed, and the numbers category came back a
plausible-looking 9/48 that was entirely an artifact of the old code. Check the
output for `Building image ...` / `Building editable for jsteer` before trusting
that a source change is live -- and where a bug would be invisible in the
results, assert the invariant inside the remote function so stale code fails
loudly instead of producing numbers (see the grader guard in ``run_causal``).

Usage
-----
::

    modal deploy modal_app.py                          # once, and after edits
    python modal_app.py warm --config qwen3-8b         # once, ~10 min
    python modal_app.py gate --config qwen3-8b         # the validation gate
    python modal_app.py rq2  --config qwen3-8b --n-shards 4
    python modal_app.py rq1  --config qwen3-8b --n-shards 8

then pull the results down and merge them locally::

    modal volume get jsteer-results / ./results
    python scripts/07_merge_shards.py
"""

from __future__ import annotations

import pathlib
import sys

import modal

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
from jsteer.config import load_config  # noqa: E402  (local, for entrypoint math)

REPO = pathlib.Path(__file__).parent
CACHE = "/cache"
RESULTS = "/results"

APP_NAME = "jsteer"
app = modal.App(APP_NAME)

cache_volume = modal.Volume.from_name("jsteer-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("jsteer-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "torch==2.13.0",
        "transformers==5.16.1",
        "huggingface_hub==1.29.0",
        "numpy==2.5.2",
        "pyyaml==6.0.3",
        "datasets==5.0.1",
        "requests==2.34.2",
        "jlens @ git+https://github.com/anthropics/jacobian-lens.git@581d398613e5602a5af361e1c34d3a92ea82ba8e",
    )
    # copy=True so the install step below can see these files at build time.
    .add_local_dir(
        REPO,
        "/root/repo",
        copy=True,
        # Dockerignore-style, so directory excludes need the trailing glob.
        # `.venv/**` is the critical one: it is several GB and would be
        # uploaded on every image rebuild. `data/**` is safe to drop because
        # the prompt sets are re-fetched in the build step below and the
        # control corpus comes from the cache volume.
        ignore=[
            "**/__pycache__/**",
            "**/*.pyc",
            ".venv/**",
            ".git/**",
            "results/**",
            "paper/**",
            "notes/**",
            "logs/**",
            "output/**",
            "tmp/**",
            "dist/**",
            "data/reference/**",
            ".claude/**",
            ".codex/**",
            ".env*",
            "PAPER DRAFT.md",
            "CLAUDE.md",
            "implementation.md",
            "research_proposal.md",
            "Gurnee et al.",
            # data/heldout ships (a few KB of frozen eval panels read by
            # path); the upstream prompt sets are re-fetched below.
            "data/corpora/**",
            "data/jlens-upstream/**",
            ".pytest_cache/**",
            ".ruff_cache/**",
            "**/*.egg-info/**",
        ],
    )
    .run_commands("cd /root/repo && pip install --no-deps -e .")
    # Bake the pinned prompt sets in rather than fetching them from GitHub in
    # every container: eight simultaneous unauthenticated fetches is a
    # rate-limit waiting to happen, and the files are a few KB.
    .run_commands(
        'cd /root/repo && python -c "'
        "from jsteer.data import fetch_experiment;"
        "fetch_experiment('flexible-generalization');"
        "fetch_experiment('probe-swap')\""
    )
    # HF_HUB_ENABLE_HF_TRANSFER is deprecated as of the hub version in this
    # image; HF_XET_HIGH_PERFORMANCE is its replacement.
    .env(
        {
            "HF_HOME": f"{CACHE}/hf",
            "HF_XET_HIGH_PERFORMANCE": "1",
            # The Jacobian sweep allocates and frees a 67 MB matrix per layer in
            # a tight loop; without this the allocator fragments and OOMs with
            # hundreds of MB nominally free.
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)

VOLUMES = {CACHE: cache_volume, RESULTS: results_volume}

#: Default Modal GPU. L40S (48 GB) is sufficient for the <=8B models and
#: cheaper; A100-80GB removes the dim_batch constraint on the 128-token
#: wikitext arm.
#:
#: Qwen3.6-27B does not fit this default: 55.6 GB of bf16 weights leaves ~24 GB
#: on an A100-80GB, which the d_model=5120 batched backward will not live
#: inside. Every entrypoint therefore takes ``--gpu``, applied per call with
#: ``.with_options`` so one image serves both scales. Use ``--gpu H200`` (141
#: GB) for the 27B.
GPU = "A100-80GB"

#: Ceiling on simultaneously running GPU containers, across every launcher.
#:
#: Nothing here bounded fan-out before: ``.map`` over N shard specs asks Modal
#: for N containers, so ``n_shards=8`` reserved eight A100-80GBs and two
#: overlapping sweeps reserved sixteen. That is how this workspace hit its
#: instance quota -- not the ~60 stopped app records, which hold no containers
#: at all. Over the quota Modal *queues* the surplus rather than failing, so
#: the symptom is a sweep that sits there, which is exactly the failure-by-
#: succeeding mode this repo keeps hitting.
#:
#: The cap costs wall-time and no GPU-seconds: eight shards at four-at-a-time
#: bill the same as eight at once and finish in two waves. Raise it only after
#: confirming the workspace quota, and pass ``--max-containers`` rather than
#: editing this, so the default stays polite for whoever runs next.
MAX_CONTAINERS = 4


def _fn(name: str, gpu: str = "", memory: int = 0, max_containers: int = 0):
    """Resolve a function on the *deployed* app, with per-call overrides.

    Every launch goes through one deployed app instead of ``modal run``. That
    is not a style preference: ``modal run`` mints a fresh **ephemeral app**
    per invocation, so each launch becomes its own dashboard entry, and about
    sixty of them accumulated under the name ``jsteer`` before anyone looked.
    Modal's API has no delete for an app record -- ``modal_proto`` exposes
    ``AppCreate``/``AppDeploy``/``AppStop`` and nothing that removes one -- so
    those entries are permanent and the only available fix is to stop making
    them. Calling a *deployed* function creates function invocations under the
    single existing app and no new app at all.

    ``with_options`` carries the GPU/memory override that used to come from
    ``.with_options`` on the local handle, so per-call tuning is unchanged.
    """
    task = _LEGACY_TASKS.get(name)
    fn = modal.Function.from_name(APP_NAME, "gpu_task" if task else name)
    # max_containers is applied unconditionally, not just when overridden: the
    # deployed default only binds calls that go through this helper, and a cap
    # that has to be remembered per launcher is a cap that will be forgotten.
    opts: dict = {"max_containers": max_containers or MAX_CONTAINERS}
    if gpu and gpu != GPU:
        opts["gpu"] = gpu
    if memory:
        opts["memory"] = memory
    fn = fn.with_options(**opts)
    return _Dispatch(fn, task) if task else fn


#: Old per-experiment function names -> the task ``gpu_task`` dispatches on.
#: Kept as a lookup so the fifteen launchers below did not all have to change
#: when nine registrations collapsed into one.
_LEGACY_TASKS = {
    "jacobian_shard": "jacobian",
    "domain_pullback_shard": "domain_pullback",
    "component_shard": "component",
    "two_hop_shard": "two_hop",
    "pullback_shard": "pullback",
    "average_n_shard": "average_n",
    "loading_shard": "loading",
    "geometry_shard": "geometry",
    "project_out_shard": "project_out",
    "ablation_shard": "ablation",
    "park_shard": "park",
    "diffmean_shard": "diffmean",
    "beta_family_shard": "beta_family",
    "horizon_shard": "horizon",
    "horizon_effect_shard": "horizon_effect",
    "fluency_shard": "fluency",
    "comp_readout_shard": "comp_readout",
    "norm_ratio": "norm_ratio",
    "causal_shard": "causal",
    "readout_shard": "readout",
    "probe_readout_shard": "probe_readout",
    "lens_prior_shard": "lens_prior",
    "energy_shard": "energy",
    "linresp_shard": "linresp",
    "precond_shard": "precond",
}


class _Dispatch:
    """Makes the one ``gpu_task`` function look like the eight it replaced.

    ``gpu_task`` takes ``(task, spec_kwargs)``, but every launcher was written
    against a function taking ``spec_kwargs`` alone. Binding the task here
    keeps ``.remote(spec)`` and ``.map(specs)`` reading exactly as before, so
    collapsing the registrations touched no launcher.
    """

    def __init__(self, fn, task: str) -> None:
        self._fn, self._task = fn, task

    def remote(self, spec_kwargs: dict) -> dict:
        return self._fn.remote(self._task, spec_kwargs)

    def spawn(self, spec_kwargs: dict):
        """Detached call: survives the client disconnecting.

        A blocking ``.remote()`` ties the call's lifetime to this process, so a
        dropped connection cancels work already running on the GPU -- which is
        how the first probe-readout attempt died, a minute in, with nothing
        wrong remotely.
        """
        return self._fn.spawn(self._task, spec_kwargs)

    def map(self, specs):
        # starmap, not map: each input is the (task, spec) pair gpu_task wants.
        return self._fn.starmap((self._task, spec) for spec in specs)


def _corpus_cache(config_name: str) -> str:
    return f"{CACHE}/corpora/fit_{config_name}.json"


# --------------------------------------------------------------------------
# cache warming
# --------------------------------------------------------------------------


@app.function(image=image, volumes=VOLUMES, timeout=3600)
def warm_cache(config_name: str, n_fit: int) -> dict:
    """Download weights, lens and the control corpus into the cache volume.

    Runs on CPU: nothing here needs an accelerator, and paying GPU rates to
    wait on a 16 GB download is the single easiest way to waste money on a
    serverless platform.
    """
    import json
    import logging
    import pathlib as pl

    import transformers
    from huggingface_hub import hf_hub_download

    from jsteer.config import load_config
    from jsteer.corpus import load_fitting_prompts
    from jsteer.loading import load_hf_model

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = load_config(config_name)

    # Through load_hf_model, not AutoModelForCausalLM directly: on multimodal
    # checkpoints the latter silently random-initialises the decoder, and
    # warming is exactly where that should be caught -- on CPU, for free,
    # before any GPU container has been billed.
    load_hf_model(config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.hf_model_id, revision=config.model_revision)
    lens_path = hf_hub_download(config.lens_repo, config.lens_filename, revision=config.lens_revision)

    import torch

    lens_meta = torch.load(lens_path, map_location="cpu", weights_only=False)
    n_prompts = getattr(lens_meta, "n_prompts", None)
    d_model = getattr(lens_meta, "d_model", None)
    del lens_meta

    cache = pl.Path(_corpus_cache(config_name))
    cache.parent.mkdir(parents=True, exist_ok=True)
    prompts = load_fitting_prompts(
        config.fit, n_fit, tokenizer=tokenizer, cache_path=cache
    )
    cache_volume.commit()
    return {
        "model": config.hf_model_id,
        "lens": lens_path,
        "lens_n_prompts": n_prompts,
        "lens_d_model": d_model,
        "n_control_prompts": len(prompts),
        "corpus_cache": str(cache),
        "sha_head": json.loads(cache.read_text())[0][:60],
    }


@app.function(image=image, volumes=VOLUMES, timeout=1800)
def inspect_lens(config_name: str) -> dict:
    """What the downloaded J_bar actually contains. CPU, seconds, no GPU.

    ``source_layers`` is the field that matters: ``resolve_layers`` intersects
    the config band with it, so a lens fitted at a coarse layer grid silently
    shrinks the sweep. Better to read it than to infer it from a log line.
    """
    from jsteer.config import load_config
    from jsteer.loading import load_lens

    config = load_config(config_name)
    lens = load_lens(config)
    band = list(config.band)
    return {
        "d_model": lens.d_model,
        "n_prompts": lens.n_prompts,
        "n_source_layers": len(lens.source_layers),
        "source_layers": list(lens.source_layers),
        "band": [band[0], band[-1]],
        "band_layers_fitted": [l for l in band if l in set(lens.source_layers)],
    }


# --------------------------------------------------------------------------
# the three sweeps
# --------------------------------------------------------------------------


# timeout=4h: on qwen3.6-27b a shard is 4 eval prompts (~24 s per layer) plus
# one 128-token wikitext control (~16x that, and the reason the fit arm is
# ~80% of this sweep's cost). At 12 layers that is ~1.6 h, which the old 2 h
# timeout would have clipped on the slow tail.
# --------------------------------------------------------------------------
# local entrypoints
# --------------------------------------------------------------------------


def warm(config: str = "qwen3-8b", n_fit: int = 100) -> None:
    """Populate the cache volume. Run once per model, before anything else."""
    print(_fn("warm_cache").remote(config, n_fit))


def gate(
    config: str = "qwen3-8b",
    swap_mode: str = "replace",
    strengths: str = "1,2",
    n_shards: int = 4,
    dim_batch: int = 128,
    limit: int = 0,
    gpu: str = GPU,
) -> None:
    """The validation gate: reproduce upstream's 76/192 and 101/192.

    Run this *before* the Jacobian sweep. It is a small fraction of that
    sweep's cost and it is what establishes whether the reconstructed swap
    operator is the one the published numbers came from -- and therefore
    whether the causal half of both research questions is answerable.
    """
    specs = [
        {
            "config_name": config,
            "mode": "swap",
            "directions": ["averaged"],
            "swap_mode": swap_mode,
            "strengths": [float(s) for s in strengths.split(",")],
            "dim_batch": dim_batch,
            "shard": i,
            "n_shards": n_shards,
            "limit": limit or None,
            "out_dir": RESULTS,
        }
        for i in range(n_shards)
    ]
    for summary in _fn("causal_shard", gpu).map(specs):
        print(summary)


def gate_grid(
    config: str = "qwen3-8b",
    bands: str = "20-23,18-25,15-28,9-34",
    strengths: str = "0.5,1,2",
    swap_mode: str = "clamp",
    dim_batch: int = 128,
    gpu: str = GPU,
) -> None:
    """Sweep the band width for the gate. Cheap, and currently the open question.

    The swap is applied at every band layer, and its effect *compounds*: each
    layer's clamp is computed from the clean residual but lands on top of every
    edit below it. Measured on qwen3-1.7b, identical trials at strength 1 give
    12/24 over a 5-layer band and 1/24 over the full 20-layer band. The bands in
    ``configs/*.yaml`` were measured on gemma-2-2b and transposed to the Qwen3
    models without being checked, so the width is a free parameter -- and the
    one most likely to explain a gate that misses 76/192 by an order of
    magnitude.

    Each band writes its own file via ``CausalSpec.label``; without that they
    would overwrite each other.
    """
    specs = []
    for band in bands.split(","):
        start, stop = (int(x) for x in band.split("-"))
        specs.append(
            {
                "config_name": config,
                "mode": "swap",
                "directions": ["averaged"],
                "swap_mode": swap_mode,
                "strengths": [float(s) for s in strengths.split(",")],
                "layers": list(range(start, stop + 1)),
                "label": f"{swap_mode}_L{start}-{stop}",
                "dim_batch": dim_batch,
                "out_dir": RESULTS,
            }
        )
    for summary in _fn("causal_shard", gpu).map(specs):
        print(summary)


def c1(
    config: str = "qwen3-8b",
    band: str = "14-19",
    mode: str = "both",
    strengths: str = "0.5,1,2,4",
    swap_mode: str = "clamp",
    n_shards: int = 4,
    dim_batch: int = 128,
    gpu: str = GPU,
    label: str = "",
    overwrite: bool = False,
) -> None:
    """C1 - direction substitution: the causal half of both research questions.

    Same prompt, same target, same operator, same magnitude -- only the
    *direction source* differs: the lens's averaged g_bar against the
    prompt-local g_x. That within-trial A/B is internally valid without any
    comparison to the paper's model, which is what makes it the experiment the
    project actually rests on.

    Band defaults to **L14-19**, which is the paper's own "first third of the
    workspace range" (L38-54 of depth) rather than a value fitted to our own
    steering scores. Fitting the band on the outcome metric would be tuning on
    the test set; this is pre-registered from the paper and happens to land
    next to our empirical optimum (L17-22) without being chosen for it.
    """
    start, stop = (int(x) for x in band.split("-"))
    specs = [
        {
            "config_name": config,
            "mode": mode,
            "directions": ["averaged", "local"],
            "swap_mode": swap_mode,
            "strengths": [float(x) for x in strengths.split(",")],
            "layers": list(range(start, stop + 1)),
            # ``label`` extends, never replaces, the band tag: two runs over
            # the same band with different strength grids must not collide.
            "label": f"c1_L{start}-{stop}" + (f"_{label}" if label else ""),
            "dim_batch": dim_batch,
            "shard": i,
            "n_shards": n_shards,
            # Re-grading an existing grid (a new metric on the same trials) is
            # the one case where rewriting a completed shard is correct. Every
            # other use of this flag is a way to lose data.
            "overwrite": overwrite,
            "out_dir": RESULTS,
        }
        for i in range(n_shards)
    ]
    for summary in _fn("causal_shard", gpu).map(specs):
        print(summary)


def rq2(
    config: str = "qwen3-8b",
    positions: str = "all",
    n_fit: int = 100,
    n_shards: int = 4,
    dim_batch: int = 128,
    limit: int = 0,
    gpu: str = GPU,
) -> None:
    """The pullback sweep. Cheap enough to run at every position convention."""
    specs = [
        {
            "config_name": config,
            "positions": positions,
            "n_fit": n_fit,
            "dim_batch": dim_batch,
            "shard": i,
            "n_shards": n_shards,
            "limit": limit or None,
            "out_dir": RESULTS,
            "corpus_cache": _corpus_cache(config),
        }
        for i in range(n_shards)
    ]
    for summary in _fn("pullback_shard", gpu).map(specs):
        print(summary)


def rq1(
    config: str = "qwen3-8b",
    positions: str = "all",
    n_fit: int = 16,
    n_shards: int = 8,
    dim_batch: int = 128,
    max_batch_tokens: int = 2048,
    k: int = 64,
    layer_chunk: int = 26,
    lowrank: bool = False,
    limit: int = 0,
    overwrite: bool = False,
    layer_stride: int = 1,
    gpu: str = GPU,
    only: str = "",
) -> None:
    """The Jacobian sweep. Run last, at the position convention rq2 argues for.

    ``n_fit`` defaults lower than the other sweeps on purpose. A prompt costs
    ``d_model * seq_len * n_layers``, so a 128-token wikitext control is ~16x an
    8-token eval prompt -- measured, 55 s against ~15 min. Thirty-two controls
    would be 8 GPU-hours to calibrate a null whose eval arm takes one.
    """
    band = load_config(config).band
    layers = list(band)[::layer_stride] if layer_stride > 1 else None
    specs = [
        {
            "config_name": config,
            "positions": positions,
            "n_fit": n_fit,
            "layers": layers,
            "dim_batch": dim_batch,
            "max_batch_tokens": max_batch_tokens,
            "k": k,
            "layer_chunk": layer_chunk,
            "lowrank": lowrank,
            "svd_device": "cuda",
            "shard": i,
            "n_shards": n_shards,
            "limit": limit or None,
            "overwrite": overwrite,
            "out_dir": RESULTS,
            "corpus_cache": _corpus_cache(config),
        }
        # ``--only 0,3`` re-runs just those shards, keeping the same
        # ``shardNof16`` tag so their outputs slot back into the same pool.
        # Needed because a shard is the unit of the running-mean sum: fixing
        # one prompt means recomputing its shard, not the whole sweep.
        for i in ([int(x) for x in only.split(",")] if only else range(n_shards))
    ]
    for summary in _fn("jacobian_shard", gpu).map(specs):
        print(summary)


#: Task name -> (spec class, runner function) in ``jsteer.sweeps``.
#:
#: These were nine separate ``@app.function``s and every one was the same five
#: lines with a different spec class substituted: build the spec from a kwargs
#: dict, call its runner, commit the volume, return the summary. Nine
#: registrations bought nothing over one dispatch, and they cluttered the app
#: with GPU-labelled entries that had never been called -- a declared default
#: reserves nothing, but a dashboard full of "A100-80GB" rows reads like it
#: does.
#:
#: ``norm_ratio_report`` is the one that differed: it returned without
#: committing the volume. That was correct rather than a bug -- it writes
#: nothing to disk, it returns its numbers -- and the extra commit it now gets
#: is a no-op.
GPU_TASKS: dict[str, tuple[str, str]] = {
    "write_controls": ("WriteControlSpec", "run_write_controls"),
    "concept": ("ConceptSpec", "run_concept"),
    "jacobian": ("JacobianSweepSpec", "run_jacobian_sweep"),
    "pullback": ("PullbackSweepSpec", "run_pullback_sweep"),
    "domain_pullback": ("DomainPullbackSpec", "run_domain_pullbacks"),
    "component": ("ComponentSpec", "run_component_pullbacks"),
    "two_hop": ("TwoHopSpec", "run_two_hop"),
    "average_n": ("AverageNSpec", "run_average_n"),
    "loading": ("AverageNSpec", "workspace_loading"),
    "norm_ratio": ("AverageNSpec", "norm_ratio_report"),
    "geometry": ("AverageNSpec", "run_direction_geometry"),
    "project_out": ("AverageNSpec", "run_project_out"),
    "beta_family": ("AverageNSpec", "run_beta_family"),
    "horizon": ("HorizonSpec", "run_horizon_pullbacks"),
    "horizon_effect": ("CausalSpec", "run_horizon_effect"),
    "fluency": ("CausalSpec", "run_fluency"),
    "comp_readout": ("AverageNSpec", "run_component_readout"),
    "ablation": ("CausalSpec", "run_ablation_effect"),
    "park": ("AverageNSpec", "run_park_whiten"),
    "diffmean": ("AverageNSpec", "run_diffmean"),
    "causal": ("CausalSpec", "run_causal"),
    "readout": ("ReadoutSpec", "run_readout"),
    "probe_readout": ("ReadoutSpec", "run_probe_readout"),
    "lens_prior": ("ReadoutSpec", "run_lens_prior"),
    "energy": ("EnergySpec", "run_energy_map"),
    "linresp": ("LinearResponseSpec", "run_linear_response"),
    "precond": ("PreconditionSpec", "run_precondition"),
}


@app.function(
    image=image,
    volumes=VOLUMES,
    gpu=GPU,
    timeout=14400,
    retries=2,
    max_containers=MAX_CONTAINERS,
)
def gpu_task(task: str, spec_kwargs: dict) -> dict:
    """Run one shard of any sweep. The single GPU entry point.

    ``timeout`` is the longest any task needs. It is a cap, not a reservation,
    so the short tasks are not charged for it. ``retries=2`` is only safe
    because every sweep is idempotent: a retried container skips the prompts
    whose outputs already landed on the volume.
    """
    import logging

    from jsteer import sweeps

    if task == "write_controls":
        from jsteer import write_controls as sweeps
    if task == "concept":
        from jsteer import concept_use as sweeps

    if task not in GPU_TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {sorted(GPU_TASKS)}")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # A WARM container holds the volume view it had when it started, so a
    # digest another task committed minutes ago is simply not there. That is
    # how the C30 causal run died: the diffmean digest was on the volume and
    # visible to ``modal volume ls``, but the containers left warm by the
    # preceding ablation run had mounted the volume before it was written.
    # Reload at the top of every task; it is cheap and the failure it prevents
    # is the one this repo keeps repeating -- a cached artifact read through a
    # stale handle.
    results_volume.reload()
    spec_cls, runner = GPU_TASKS[task]
    summary = getattr(sweeps, runner)(getattr(sweeps, spec_cls)(**spec_kwargs))
    results_volume.commit()
    return summary


@app.function(
    image=image,
    volumes=VOLUMES,
    gpu=GPU,
    timeout=3600,
    memory=65536,
    max_containers=MAX_CONTAINERS,
)
def pool_shard_means(
    config_name: str,
    positions: str,
    k: int,
    shards: int,
    namespace: str = "",
    save_means: bool = False,
) -> dict:
    """Pool the running sums on Modal; only the digest comes home.

    ``memory=65536`` because pooling holds the accumulator (~7 GB) plus one
    shard file (~7 GB) at a time, and torch.load is all-or-nothing per file.

    ``namespace`` selects which run to pool -- the domain panel writes under
    its own, so that its sums are never pooled together with the
    eval-vs-wikitext sweep's. ``save_means`` additionally writes the full mean
    matrices, which the rank-k digest cannot substitute for: reading one
    domain's activations through another domain's lens needs the whole
    ``J_bar_domain``, not its top-k subspace.
    """
    import logging

    from jsteer.sweeps import pool_group_means

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = pool_group_means(
        config_name,
        positions=positions,
        namespace=namespace or None,
        k=k,
        shards=shards or None,
        out_dir=RESULTS,
        svd_device="cuda",
        save_means=save_means,
    )
    results_volume.commit()
    return summary


def pool(
    config: str = "qwen3-8b",
    positions: str = "all",
    k: int = 64,
    shards: int = 0,
    gpu: str = GPU,
    memory: int = 0,
) -> None:
    """Reduce the per-shard fp64 sums to a ~125 MB digest, in place.

    The accumulator is ``d_model^2`` fp64 per (group, layer): 210 MB a piece at
    d_model=5120, so a 35-layer two-group pool holds ~15 GB and torch.load
    brings another ~15 GB in beside it. Pass ``--memory 131072`` for the 27B.
    """
    print(
        _fn("pool_shard_means", gpu, memory).remote(
            config, positions, k, shards, "", False
        )
    )


def smoke27(config: str = "qwen3.6-27b", gpu: str = "H200") -> None:
    """The 27B-specific smoke test. Run before anything expensive.

    Three things are unverified on this model and all three are cheap to check:

    1. **Does the checkpoint load into a usable model at all?** The load guard
       covers key coverage; a forward pass covers whether the multimodal
       wrapper's text path actually works headlessly. The causal arm's clean
       baseline is the readout -- a random 27B scores ~0/192.
    2. **Does autograd survive the gated-linear-attention blocks?** Three of
       every four layers here are not softmax attention. jlens takes a VJP
       through them; transformers' native path is differentiable but the fused
       kernels are not, and this is where that would surface.
    3. **What does a (prompt, layer) Jacobian cost?** The full RQ1 sweep is
       35 band layers x 5120 backward passes x 80 prompts. Extrapolating that
       from three measured layers is the difference between a $30 run and an
       unpleasant surprise.
    """
    print(_fn("inspect_lens").remote(config))
    print(
        _fn("causal_shard", gpu).remote(
            {
                "config_name": config,
                "mode": "swap",
                "directions": ["averaged"],
                "strengths": [1.0],
                "layers": [24, 29, 34],
                "limit": 12,
                "label": "smoke27",
                "dim_batch": 32,
                "out_dir": RESULTS,
            }
        )
    )
    print(
        _fn("jacobian_shard", gpu).remote(
            {
                "config_name": config,
                "n_fit": 0,
                "limit": 2,
                "layers": [24, 41, 58],
                "dim_batch": 64,
                "max_batch_tokens": 1024,
                "k": 32,
                "layer_chunk": 3,
                "svd_device": "cuda",
                "out_dir": RESULTS,
                "corpus_cache": _corpus_cache(config),
            }
        )
    )


def smoke(config: str = "qwen3-1.7b") -> None:
    """End-to-end infrastructure check on the small model, for a few cents.

    Exercises image build, both volume mounts, the corpus cache, sharding, and
    result writes. Every failure this catches is an infrastructure failure, not
    a scientific one -- which is exactly why it should run before the 8B model
    is ever loaded.
    """
    print(_fn("warm_cache").remote(config, 4))
    print(
        _fn("causal_shard").remote(
            {
                "config_name": config,
                "mode": "both",
                "directions": ["averaged", "local"],
                "strengths": [1.0],
                "layers": [13, 17, 21],
                "limit": 6,
                "out_dir": RESULTS,
            }
        )
    )
    print(
        _fn("pullback_shard").remote(
            {
                "config_name": config,
                "n_fit": 2,
                "limit": 6,
                "layers": [13, 17, 21],
                "out_dir": RESULTS,
                "corpus_cache": _corpus_cache(config),
            }
        )
    )
    print(
        _fn("jacobian_shard").remote(
            {
                "config_name": config,
                "n_fit": 1,
                "limit": 3,
                "layers": [13, 17, 21],
                "dim_batch": 64,
                "k": 32,
                "layer_chunk": 3,
                "svd_device": "cuda",
                "out_dir": RESULTS,
                "corpus_cache": _corpus_cache(config),
            }
        )
    )


def lens(config: str = "qwen3-8b") -> None:
    """Print the downloaded lens's shape, fit count and fitted layer grid."""
    print(_fn("inspect_lens").remote(config))


def c4(
    config: str = "qwen3.6-27b",
    band: str = "24-34",
    ns: str = "1,2,5,10,25,50,100",
    n_pool: int = 100,
    replicates: int = 5,
    strength: float = 0.5,
    n_shards: int = 4,
    dim_batch: int = 128,
    gpu: str = GPU,
    label: str = "",
    overwrite: bool = False,
) -> None:
    """C4 - does the averaged direction's advantage come from sample size?

    Steers with the mean of g_x over n wikitext prompts, for a ladder of n, and
    compares against the lens's own J_bar and against two n=1 controls (the
    prompt's own g_x and another prompt's). A smooth climb means averaging is
    variance reduction; a flat curve or a plateau below J_bar means it is not.

    Uses the swap operator only, because swap is magnitude-invariant and one
    strength is therefore fair across every n -- see AverageNSpec.
    """
    start, stop = (int(x) for x in band.split("-"))
    specs = [
        {
            "config_name": config,
            "layers": list(range(start, stop + 1)),
            "ns": [int(x) for x in ns.split(",")],
            "n_pool": n_pool,
            "replicates": replicates,
            "strength": strength,
            "label": label or f"L{start}-{stop}",
            "dim_batch": dim_batch,
            "shard": i,
            "n_shards": n_shards,
            "overwrite": overwrite,
            "out_dir": RESULTS,
            "corpus_cache": _corpus_cache(config),
        }
        for i in range(n_shards)
    ]
    for summary in _fn("average_n_shard", gpu).map(specs):
        print(summary)


def c4_norms(
    config: str = "qwen3.6-27b",
    band: str = "24-34",
    ns: str = "1,2,5,10,25,50,100",
    n_pool: int = 100,
    replicates: int = 5,
    gpu: str = GPU,
) -> None:
    """Is the C4 curve confounded by the swap's column-norm ratio drifting?

    The swap is invariant to scaling V uniformly but NOT to scaling its two
    columns differently, so C4's single-strength design is only sound if
    ||v_s||/||v_t|| is stable across n. This measures it.
    """
    start, stop = (int(x) for x in band.split("-"))
    print(
        _fn("norm_ratio", gpu).remote(
            {
                "config_name": config,
                "layers": list(range(start, stop + 1)),
                "ns": [int(x) for x in ns.split(",")],
                "n_pool": n_pool,
                "replicates": replicates,
                "corpus_cache": _corpus_cache(config),
            }
        )
    )


def comp_readout(
    config: str = "qwen3-8b",
    band: str = "13-31",
    layer_stride: int = 3,
    directions: str = "",
    arms: str = "",
    label: str = "comp",
    gpu: str = GPU,
) -> None:
    """C24 - the read side of the component split, on our own lens.

    Restricted-vocabulary: ranks the prompt's argument among the 16 fitted
    target words, not the full vocabulary, because the component sweep only
    materialized per-word pullbacks. Comparable across arms, NOT comparable to
    C14's pass@k.
    """
    start, stop = (int(x) for x in band.split("-"))
    print(
        _fn("comp_readout_shard", gpu).remote(
            {
                "config_name": config,
                "layers": list(range(start, stop + 1))[::layer_stride],
                "out_dir": RESULTS,
                "label": label,
                "directions_paths": [
                    f"{RESULTS}/rq1/group_mean_digests_{ns.strip()}.pt"
                    for ns in directions.split(",")
                    if ns.strip()
                ],
                "arms": [a.strip() for a in arms.split(",") if a.strip()],
            }
        )
    )


def fluency(
    config: str = "qwen3-8b",
    namespaces: str = "qwen3-8b_dom2x32_968c0ba0_pb_comp",
    arms: str = "averaged,gsm8k_full,gsm8k_off,wikitext_a_full,wikitext_a_off",
    strengths: str = "1.0",
    layer_stride: int = 3,
    gen_tokens: int = 32,
    limit: int = 48,
    n_shards: int = 2,
    label: str = "comp",
    gpu: str = GPU,
) -> None:
    """C23 - does dropping the diagonal term damage generation?

    Same swap as ``domain_causal`` at the same layers, but decodes 32 tokens
    instead of 4 and scores them for clean-model NLL, repeated 4-grams and
    longest repeated-token run. ``limit`` keeps this tractable: generation is
    O(tokens) forward passes per arm per trial, so the full 192-trial grid at
    eleven arms would be ~200k passes.
    """
    band = load_config(config).band
    specs = [
        {
            "config_name": config,
            "mode": "swap",
            "strengths": [float(x) for x in strengths.split(",")],
            "directions": [a.strip() for a in arms.split(",") if a.strip()],
            "directions_paths": [
                f"{RESULTS}/rq1/group_mean_digests_{ns.strip()}.pt"
                for ns in namespaces.split(",")
                if ns.strip()
            ],
            "layers": list(band)[::layer_stride] if layer_stride > 1 else None,
            "gen_tokens": gen_tokens,
            "limit": limit or None,
            "label": label,
            "shard": i,
            "n_shards": n_shards,
            "out_dir": RESULTS,
        }
        for i in range(n_shards)
    ]
    for summary in _fn("fluency_shard", gpu).map(specs):
        print(summary)


def horizon_effect(
    config: str = "qwen3-8b",
    namespaces: str = "",
    arms: str = "",
    strength: float = 1.0,
    layer_stride: int = 3,
    gen_tokens: int = 16,
    limit: int = 96,
    n_shards: int = 2,
    label: str = "hzfx",
    gpu: str = GPU,
) -> None:
    """C27 - does writing at horizon d move the output d tokens later?

    Writes ``v_y^(d)`` additively at the prompt positions and measures the
    log-prob lift of ``y`` at each of the next ``gen_tokens`` positions, with
    the continuation teacher-forced to the clean greedy text so position k
    means the same thing for every arm.
    """
    band = load_config(config).band
    specs = [
        {
            "config_name": config,
            "mode": "additive",
            "strengths": [strength],
            "directions": [a.strip() for a in arms.split(",") if a.strip()],
            "directions_paths": [
                f"{RESULTS}/rq1/group_mean_digests_{ns.strip()}.pt"
                for ns in namespaces.split(",")
                if ns.strip()
            ],
            "layers": list(band)[::layer_stride] if layer_stride > 1 else None,
            "gen_tokens": gen_tokens,
            "limit": limit or None,
            "label": label,
            "shard": i,
            "n_shards": n_shards,
            "out_dir": RESULTS,
        }
        for i in range(n_shards)
    ]
    for summary in _fn("horizon_effect_shard", gpu).map(specs):
        print(summary)


def horizon(
    config: str = "qwen3-8b",
    corpora: str = "wikitext_a",
    n_per: int = 12,
    layer_stride: int = 3,
    t_stride: int = 1,
    dim_batch: int = 0,
    gpu: str = GPU,
) -> None:
    """C26 gating check - is the off-diagonal term horizon-agnostic?

    Bins the pullback by d = t' - t and reports the pairwise cosine between
    horizon buckets. Near-1 everywhere means the only special horizon is 0 and
    horizon-targeted writing has nothing to address. Deliberately a small panel
    -- it costs what the component sweep costs per prompt.
    """
    band = load_config(config).band
    spec = {
        "config_name": config,
        "domains": [c.strip() for c in corpora.split(",") if c.strip()],
        "n_per": n_per,
        "layers": list(band)[::layer_stride] if layer_stride > 1 else None,
        "out_dir": RESULTS,
        "corpus_cache": f"{RESULTS}/energy_corpora.json",
        "t_stride": t_stride,
        "dim_batch": dim_batch or None,
    }
    call = _fn("horizon_shard", gpu).spawn(spec)
    print(f"spawned {call.object_id}; waiting (safe to disconnect)")
    print(call.get())


def beta_family(
    config: str = "qwen3-8b",
    band: str = "13-31",
    layer_stride: int = 3,
    directions: str = "",
    betas: str = "0,0.5,1,2,3",
    label: str = "beta",
    gpu: str = GPU,
) -> None:
    """C25 - sweep the unembedding correction, and build its closed-form surrogate.

    ``v_y(beta) = v_y - beta * <v_y, u_hat_y> * u_hat_y``; beta=0 is the
    standard lens, beta=1 is C22's ``_perp``, beta>1 over-corrects. Also emits
    ``_galpha``, which subtracts one scalar multiple of ``u_y`` per layer and
    so needs no per-position fit at all.
    """
    start, stop = (int(x) for x in band.split("-"))
    print(
        _fn("beta_family_shard", gpu).remote(
            {
                "config_name": config,
                "layers": list(range(start, stop + 1))[::layer_stride],
                "out_dir": RESULTS,
                "label": label,
                "ns": [float(b) for b in betas.split(",")],
                "directions_paths": [
                    f"{RESULTS}/rq1/group_mean_digests_{ns.strip()}.pt"
                    for ns in directions.split(",")
                    if ns.strip()
                ],
            }
        )
    )


def project_out(
    config: str = "qwen3-8b",
    band: str = "13-31",
    layer_stride: int = 3,
    directions: str = "",
    arms: str = "",
    label: str = "perp",
    gpu: str = GPU,
) -> None:
    """C22 - the Gram-Schmidt control C21 asks for.

    Removes the token's own unembedding row from the FULL write direction and
    leaves the diagonal otherwise intact, plus a random-donor control that
    removes an equal-magnitude component along a different token's row. Emits
    a causal-ready digest, so ``domain_causal --namespaces`` steers it next.
    """
    start, stop = (int(x) for x in band.split("-"))
    print(
        _fn("project_out_shard", gpu).remote(
            {
                "config_name": config,
                "layers": list(range(start, stop + 1))[::layer_stride],
                "out_dir": RESULTS,
                "label": label,
                "directions_paths": [
                    f"{RESULTS}/rq1/group_mean_digests_{ns.strip()}.pt"
                    for ns in directions.split(",")
                    if ns.strip()
                ],
                "arms": [a.strip() for a in arms.split(",") if a.strip()],
            }
        )
    )


def ablation(
    config: str = "qwen3-8b",
    namespaces: str = "",
    arms: str = "",
    layer_stride: int = 3,
    n_shards: int = 2,
    label: str = "abl",
    limit: int = 0,
    overwrite: bool = False,
    gpu: str = GPU,
) -> None:
    """C28 - score our arms on the workspace paper's OWN causal measure.

    Deletes each write direction from the residual stream and records the
    induced output KL (their App. A.6/A.7 convention: higher = more causally
    important), on the same trials and the same band as the swap runs. The
    thesis predicts the arms are flat here and far apart on swap success.

    Two forward passes per (trial, arm), no backward passes and no generation,
    so this is minutes.
    """
    band = load_config(config).band
    layers = list(band)[::layer_stride] if layer_stride > 1 else None
    paths = [
        f"{RESULTS}/rq1/group_mean_digests_{ns.strip()}.pt"
        for ns in namespaces.split(",")
        if ns.strip()
    ]
    print("directions from:", paths)
    specs = [
        {
            "config_name": config,
            "directions": [a.strip() for a in arms.split(",") if a.strip()],
            "directions_paths": paths,
            "layers": layers,
            "label": label,
            "shard": i,
            "n_shards": n_shards,
            "limit": limit or None,
            "overwrite": overwrite,
            "out_dir": RESULTS,
        }
        for i in range(n_shards)
    ]
    for summary in _fn("ablation_shard", gpu).map(specs):
        print(summary)


def park(
    config: str = "qwen3-8b",
    band: str = "13-31",
    layer_stride: int = 3,
    directions: str = "",
    arms: str = "",
    label: str = "park",
    gpu: str = GPU,
) -> None:
    """C29 - Park et al. 2023's causal-inner-product whitening, as a baseline.

    Emits ``Cov(gamma)^-1 v`` and ``Cov(gamma)^-1/2 v`` for every loaded arm,
    at two ridges, plus the Jacobian-free ``uy`` / ``uy_park`` arms. This is
    the baseline that can collapse our contribution into prior work, so it is
    run before anything is claimed. Writes a causal-ready digest for
    ``domain_causal`` to steer.
    """
    start, stop = (int(x) for x in band.split("-"))
    print(
        _fn("park_shard", gpu).remote(
            {
                "config_name": config,
                "layers": list(range(start, stop + 1))[::layer_stride],
                "out_dir": RESULTS,
                "label": label,
                "directions_paths": [
                    f"{RESULTS}/rq1/group_mean_digests_{ns.strip()}.pt"
                    for ns in directions.split(",")
                    if ns.strip()
                ],
                "arms": [a.strip() for a in arms.split(",") if a.strip()],
            }
        )
    )


def diffmean(
    config: str = "qwen3-8b",
    band: str = "13-31",
    layer_stride: int = 3,
    n_fit: int = 2,
    label: str = "dm",
    gpu: str = GPU,
) -> None:
    """C30 - DiffMean/CAA, the supervised per-concept baseline.

    Contrast pairs come from the dataset's own structure: same category, same
    template, different argument. ``n_fit`` templates per category are used for
    the fit and the rest are held out, so the causal run downstream can be
    scored separately on seen and unseen templates -- the seen half is a
    ceiling, the unseen half is the fair number.

    64 forward passes, no backward passes. Minutes.
    """
    start, stop = (int(x) for x in band.split("-"))
    print(
        _fn("diffmean_shard", gpu).remote(
            {
                "config_name": config,
                "layers": list(range(start, stop + 1))[::layer_stride],
                "out_dir": RESULTS,
                "label": label,
                "n_fit": n_fit,
            }
        )
    )


def geometry(
    config: str = "qwen3-8b",
    band: str = "13-31",
    layer_stride: int = 3,
    directions: str = "",
    arms: str = "",
    label: str = "",
    gpu: str = GPU,
) -> None:
    """C21 - is the diagonal half of the write direction just the unembedding row?

    Tests the cheapest mechanism for C18: the residual stream is a sum, so the
    ``t' = t`` block carries an identity path, and if it dominates then
    ``J_diag^T u_y ~ u_y`` -- i.e. the short-horizon half of a J-lens write is
    approximately logit-lens steering, and ``_off`` is what remains once that
    is removed.

    Linear algebra over a cached digest: no prompts and no backward passes, so
    this is a couple of minutes and is dominated by loading the model for its
    unembedding rows.
    """
    start, stop = (int(x) for x in band.split("-"))
    print(
        _fn("geometry_shard", gpu).remote(
            {
                "config_name": config,
                "layers": list(range(start, stop + 1))[::layer_stride],
                "out_dir": RESULTS,
                "label": label,
                "directions_paths": [
                    f"{RESULTS}/rq1/group_mean_digests_{ns.strip()}.pt"
                    for ns in directions.split(",")
                    if ns.strip()
                ],
                "arms": [a.strip() for a in arms.split(",") if a.strip()],
            }
        )
    )


def loading(
    config: str = "qwen3.6-27b",
    band: str = "24-34",
    layer_stride: int = 1,
    directions: str = "",
    arms: str = "",
    gpu: str = GPU,
) -> None:
    """Compute workspace loading -- the J-Lens paper's own predictor of swap success.

    Section 3.4: "the cosine similarity between the residual stream and that
    concept's lens vector, averaged over the argument and readout positions in
    the unmodified forward pass", for the SOURCE argument. Our mentor asked for
    it as a baseline against the geometric predictors; it is citable rather
    than invented, and it needs one clean forward pass per prompt.
    """
    start, stop = (int(x) for x in band.split("-"))
    print(
        _fn("loading_shard", gpu).remote(
            {
                "config_name": config,
                "layers": list(range(start, stop + 1))[::layer_stride],
                "out_dir": RESULTS,
                "corpus_cache": _corpus_cache(config),
                "directions_paths": [
                    f"{RESULTS}/rq1/group_mean_digests_{ns.strip()}.pt"
                    for ns in directions.split(",")
                    if ns.strip()
                ],
                "arms": [a.strip() for a in arms.split(",") if a.strip()],
            }
        )
    )


def c5(
    config: str = "qwen3-8b",
    band: str = "14-19",
    alphas: str = "0.001,0.003,0.01,0.03,0.1,0.3,1,2,4",
    n_random: int = 2,
    n_shards: int = 4,
    dtype: str = "float32",
    dim_batch: int = 16,
    limit: int = 0,
    gpu: str = GPU,
    label: str = "",
    overwrite: bool = False,
) -> None:
    """C5 - which Jacobian is the better *local model*, and out to what dose?

    Perturbs one layer at one position and compares the measured change in
    ``<u_y, h_final[last]>`` against what J_x and J_bar each predict, over a
    dose grid running four orders of magnitude below the steering regime.

    Two things come out of it. It is the validation we never ran -- if J_x does
    not out-predict J_bar at the smallest dose, our Jacobian is wrong and every
    negative causal result is uninterpretable. And if the two error curves
    cross, the crossover locates the dose above which the local Jacobian stops
    being the better model, which is RQ3 answered on its own terms rather than
    through an argmax.

    Runs in **float32**, not the config's bfloat16: the measurement is a
    difference of two forward passes and bf16 quantizes the small-alpha end
    away. That costs ~32 GB of weights for the 8B, so keep the 80 GB card.
    """
    start, stop = (int(x) for x in band.split("-"))
    specs = [
        {
            "config_name": config,
            "layers": list(range(start, stop + 1)),
            "alphas": [float(x) for x in alphas.split(",")],
            "n_random": n_random,
            "dtype": dtype,
            "dim_batch": dim_batch,
            "label": label or f"L{start}-{stop}",
            "limit": limit or None,
            "shard": i,
            "n_shards": n_shards,
            "overwrite": overwrite,
            "out_dir": RESULTS,
        }
        for i in range(n_shards)
    ]
    for summary in _fn("linresp_shard", gpu).map(specs):
        print(summary)


def c7(
    config: str = "qwen3-8b",
    band: str = "14-19",
    mode: str = "both",
    strengths: str = "0.003,0.01,0.03,0.1,0.3,1,2",
    rungs: str = "none,sigma,gn,fisher",
    directions: str = "averaged,local",
    tau: float = 0.1,
    n_cov: int = 100,
    k: int = 64,
    n_shards: int = 8,
    dim_batch: int = 128,
    limit: int = 0,
    gpu: str = GPU,
    label: str = "",
    overwrite: bool = False,
) -> None:
    """C7 - does raising the index rescue the prompt-local direction?

    Every steering arm before this one wrote the raw pullback, which is
    FishBack's G = I case -- so our negative result about prompt-local
    directions is a result about the Euclidean metric and nothing else. This
    runs the same trials with the direction preconditioned by the activation
    covariance, by J^T J, and by the pullback Fisher J^T H J, on both the local
    and the averaged direction.

    Needs the RQ1 rank-k SVD digests on the results volume for the gn and
    fisher rungs; sigma needs only the fitting corpus.
    """
    start, stop = (int(x) for x in band.split("-"))
    specs = [
        {
            "config_name": config,
            "mode": mode,
            "layers": list(range(start, stop + 1)),
            "strengths": [float(x) for x in strengths.split(",")],
            "preconditioners": rungs.split(","),
            "directions": directions.split(","),
            "tau": tau,
            "n_cov": n_cov,
            "k": k,
            "label": label or f"L{start}-{stop}",
            "dim_batch": dim_batch,
            "limit": limit or None,
            "shard": i,
            "n_shards": n_shards,
            "overwrite": overwrite,
            "out_dir": RESULTS,
            "corpus_cache": _corpus_cache(config),
        }
        for i in range(n_shards)
    ]
    for summary in _fn("precond_shard", gpu).map(specs):
        print(summary)


# --------------------------------------------------------------------------
# the domain panel
# --------------------------------------------------------------------------


def _domain_spec(
    config: str,
    corpora: str,
    n_per: int,
    min_tokens: int,
    layer_stride: int,
    **rest,
) -> dict:
    """Build one domain-sweep spec dict, and derive the paths that depend on it."""
    from jsteer.sweeps import JacobianSweepSpec

    band = load_config(config).band
    names = [name.strip() for name in corpora.split(",") if name.strip()]
    probe = JacobianSweepSpec(
        config_name=config, domains=names, n_per=n_per, min_tokens=min_tokens
    )
    return {
        "config_name": config,
        "domains": names,
        "n_per": n_per,
        "min_tokens": min_tokens,
        "layers": list(band)[::layer_stride] if layer_stride > 1 else None,
        # One materialized panel on the cache volume, keyed by the panel's own
        # identity, so all shards stride the same document list.
        "corpus_cache": f"{CACHE}/corpora/{probe.namespace}.json",
        "out_dir": RESULTS,
        "svd_device": "cuda",
        **rest,
    }


def domains(
    config: str = "qwen3-8b",
    corpora: str = "wikitext_a,wikitext_b,wikitext_c,wikitext_d,openwebmath,ultrachat",
    n_per: int = 16,
    min_tokens: int = 128,
    n_shards: int = 8,
    layer_stride: int = 3,
    dim_batch: int = 128,
    max_batch_tokens: int = 2048,
    k: int = 64,
    limit: int = 0,
    overwrite: bool = False,
    gpu: str = GPU,
    only: str = "",
) -> None:
    """Domain-conditional averaged Jacobians: one J_bar per corpus.

    Budget. A full ``J_x`` on a 128-token prompt is ~15 min on qwen3-8b, so the
    default panel (4 corpora x 16) is ~16 GPU-hours, ~2 h wall across 8 shards.

    ``layer_stride`` defaults to 3 -- seven layers spanning the band rather
    than twenty. That is a *storage* decision, not a compute one: one backward
    pass yields the gradient at every source layer at once, so the layer count
    barely moves the GPU time, while each (group, layer) running sum is a
    134 MB fp64 matrix. Twenty layers would be 10.7 GB per shard file and
    86 GB for the run; seven is 3.75 GB and 30 GB.
    """
    specs = [
        _domain_spec(
            config,
            corpora,
            n_per,
            min_tokens,
            layer_stride,
            dim_batch=dim_batch,
            max_batch_tokens=max_batch_tokens,
            k=k,
            shard=i,
            n_shards=n_shards,
            limit=limit or None,
            overwrite=overwrite,
        )
        for i in ([int(x) for x in only.split(",")] if only else range(n_shards))
    ]
    print(f"panel: {specs[0]['domains']}, layers {specs[0]['layers']}")
    for summary in _fn("jacobian_shard", gpu).map(specs):
        print(summary)


def pool_domains(
    config: str = "qwen3-8b",
    corpora: str = "wikitext_a,wikitext_b,wikitext_c,wikitext_d,openwebmath,ultrachat",
    n_per: int = 16,
    min_tokens: int = 128,
    shards: int = 8,
    k: int = 64,
    memory: int = 65536,
    gpu: str = "",
) -> None:
    """Pool the domain sums into one J_bar per corpus, keeping the full matrices.

    ``save_means`` is on: the cross-domain questions ("read one corpus's
    activations through another corpus's lens") need the whole matrix, and the
    rank-k digest cannot answer them.
    """
    from jsteer.sweeps import JacobianSweepSpec

    names = [name.strip() for name in corpora.split(",") if name.strip()]
    probe = JacobianSweepSpec(
        config_name=config, domains=names, n_per=n_per, min_tokens=min_tokens
    )
    print(f"pooling namespace {probe.namespace}")
    print(
        _fn("pool_shard_means", gpu, memory).remote(
            config, "all", k, shards, probe.namespace, True
        )
    )


def domain_causal(
    config: str = "qwen3-8b",
    namespaces: str = "qwen3-8b_dom6x32_7e3215f2,qwen3-8b_dom2x32_57462165",
    arms: str = "averaged,wikitext_a,openwebmath,gsm8k,arith_words",
    strengths: str = "1.0,2.0",
    layer_stride: int = 3,
    n_shards: int = 4,
    label: str = "domain",
    limit: int = 0,
    overwrite: bool = False,
    gpu: str = GPU,
) -> None:
    """Steer the 192 flexible-generalization trials with domain-averaged directions.

    Every arm is run at the *same* seven layers the domain Jacobians were
    fitted at (band, stride 3), including ``averaged`` -- the published lens.
    That makes the published arm the internal reference for whether a sparse
    seven-layer band steers at all, and it is why these numbers are not
    comparable to the earlier contiguous-band runs.

    ``namespaces`` names the pooled runs to read directions from, **literally**
    rather than by re-deriving them from a DomainSpec. The namespace is a hash
    of the corpus definitions, so extending ``DomainSpec`` with a new field
    changes what today's code derives while the artifact on disk keeps the name
    it was written under. Deriving here would then point at a file that does
    not exist -- which is what happened, and which the content-addressing is
    supposed to cause rather than prevent. The artifact's identity is its
    stored name; address it directly.
    """
    band = load_config(config).band
    layers = list(band)[::layer_stride] if layer_stride > 1 else None

    paths = [
        f"{RESULTS}/rq1/group_mean_digests_{ns.strip()}.pt"
        for ns in namespaces.split(",")
        if ns.strip()
    ]
    print("directions from:", paths)

    specs = [
        {
            "config_name": config,
            "mode": "swap",
            "strengths": [float(x) for x in strengths.split(",")],
            "directions": [a.strip() for a in arms.split(",") if a.strip()],
            "directions_paths": paths,
            "layers": layers,
            "label": label,
            "shard": i,
            "n_shards": n_shards,
            "limit": limit or None,
            "overwrite": overwrite,
            "out_dir": RESULTS,
        }
        for i in range(n_shards)
    ]
    for summary in _fn("causal_shard", gpu).map(specs):
        print(summary)


# --------------------------------------------------------------------------
# the launcher
# --------------------------------------------------------------------------


#: Every function above that is a launch, in listing order. These used to be
#: ``@app.local_entrypoint()`` and be invoked with ``modal run``; they are now
#: plain local functions dispatched by :func:`main`. See :func:`_fn` for why.
def readout(
    config: str = "qwen3-8b",
    means: str = (
        f"{RESULTS}/rq1/group_means_full_qwen3-8b_dom2x32_57462165.pt,"
        f"{RESULTS}/rq1/group_means_full_qwen3-8b_dom6x32_7e3215f2.pt"
    ),
    groups: str = "gsm8k,wikitext_a",
    layer_stride: int = 3,
    permute_acts: int = 0,
    gpu: str = GPU,
    memory: int = 131072,
    overwrite: bool = False,
) -> None:
    """Does a domain-averaged J_bar READ better, or only write better?

    No Jacobians are computed: the full mean matrices already exist on the
    volume from the C8 panels, and this only reads them. One forward pass per
    eval prompt, then matmuls -- minutes, not hours.

    ``memory`` is large because the two panels hold 2 and 6 corpora at 7 layers
    each, and a d_model^2 fp32 mean is 67 MB; loading both files is ~3.7 GB
    before the unwanted groups are dropped.
    """
    band = load_config(config).band
    spec = {
        "config_name": config,
        "means_paths": [m for m in means.split(",") if m],
        "groups": [g.strip() for g in groups.split(",") if g.strip()],
        "layers": list(band)[::layer_stride] if layer_stride > 1 else None,
        "out_dir": RESULTS,
        "overwrite": overwrite,
        "permute_acts": permute_acts,
    }
    call = _fn("readout_shard", gpu, memory).spawn(spec)
    print(f"spawned {call.object_id}; waiting (safe to disconnect)")
    print(call.get())


def probe_readout(
    config: str = "qwen3-8b",
    means: str = (
        f"{RESULTS}/rq1/group_means_full_qwen3-8b_dom2x32_57462165.pt,"
        f"{RESULTS}/rq1/group_means_full_qwen3-8b_dom6x32_7e3215f2.pt"
    ),
    groups: str = "gsm8k,wikitext_a",
    layer_stride: int = 3,
    permute_acts: int = 0,
    gpu: str = GPU,
    memory: int = 131072,
    overwrite: bool = False,
) -> None:
    """Readout of a LATENT intermediate that the prompt never states.

    The stronger readout test: probe-swap's ``intermediate`` (Brazil, for "the
    country where the Amazon River ends") appears nowhere in the input. Scores
    it alongside the answer and a logit-lens baseline.
    """
    band = load_config(config).band
    spec = {
        "config_name": config,
        "means_paths": [m for m in means.split(",") if m],
        "groups": [g.strip() for g in groups.split(",") if g.strip()],
        "layers": list(band)[::layer_stride] if layer_stride > 1 else None,
        "out_dir": RESULTS,
        "overwrite": overwrite,
        "permute_acts": permute_acts,
    }
    # spawn, not remote: a blocking .remote() ties the call's life to this
    # client, and a dropped connection cancels work that is already running on
    # the GPU. That happened here -- the first attempt died to "Received a
    # cancellation signal" a minute in, with nothing wrong on the remote side.
    call = _fn("probe_readout_shard", gpu, memory).spawn(spec)
    print(f"spawned {call.object_id}; waiting (safe to disconnect)")
    print(call.get())


def lens_prior(
    config: str = "qwen3-8b",
    means: str = (
        f"{RESULTS}/rq1/group_means_full_qwen3-8b_dom2x32_57462165.pt,"
        f"{RESULTS}/rq1/group_means_full_qwen3-8b_dom6x32_7e3215f2.pt"
    ),
    groups: str = "gsm8k,wikitext_a",
    layer_stride: int = 3,
    gpu: str = GPU,
    memory: int = 131072,
) -> None:
    """Which token classes does a corpus inflate? The mechanism diagnostic.

    Reads only the stored means and W_U -- no eval forward passes at all, so
    this is the cheapest thing in the app. Its job is to say whether a corpus's
    readout advantage is carried by a few token classes (coverage) or is broad.
    """
    band = load_config(config).band
    spec = {
        "config_name": config,
        "means_paths": [m for m in means.split(",") if m],
        "groups": [g.strip() for g in groups.split(",") if g.strip()],
        "layers": list(band)[::layer_stride] if layer_stride > 1 else None,
        "out_dir": RESULTS,
    }
    print(_fn("lens_prior_shard", gpu, memory).remote(spec))


def energy(
    config: str = "qwen3-8b",
    corpora: str = "gsm8k,wikitext_a,openwebmath,arith_words",
    n_docs: int = 24,
    n_probe: int = 8,
    t_stride: int = 8,
    layer_stride: int = 3,
    gpu: str = GPU,
    memory: int = 32768,
) -> None:
    """Where does each corpus put its Jacobian energy? The mechanism arm.

    Tests the hypothesis that GSM8K's advantage is short-horizon structure in
    the sense of Yan et al. Sec. 3.3 -- energy on the t~t' diagonal -- rather
    than anything about its subject matter, which seven content ablations
    already failed to pin down.
    """
    band = load_config(config).band
    spec = {
        "config_name": config,
        "corpora": [c.strip() for c in corpora.split(",") if c.strip()],
        "n_docs": n_docs,
        "n_probe": n_probe,
        "t_stride": t_stride,
        "layers": list(band)[::layer_stride] if layer_stride > 1 else None,
        "out_dir": RESULTS,
        "corpus_cache": f"{RESULTS}/energy_corpora.json",
    }
    call = _fn("energy_shard", gpu, memory).spawn(spec)
    print(f"spawned {call.object_id}; waiting (safe to disconnect)")
    print(call.get())


def components(
    config: str = "qwen3-8b",
    corpora: str = "gsm8k,wikitext_a",
    n_per: int = 32,
    layer_stride: int = 3,
    limit: int = 0,
    words: str = "",
    t_stride: int = 1,
    dim_batch: int = 0,
    gpu: str = GPU,
) -> None:
    """Write directions for each half of Yan et al.'s Eq. 20 split.

    Emits ``{corpus}_full``, ``{corpus}_diag`` and ``{corpus}_off`` into one
    causal-ready digest, so ``domain_causal --arms`` steers them unchanged.
    ``_full`` must reproduce the plain domain digest; it is the baseline the
    halves are read against.

    ~111 backward passes per prompt against the usual one, so 2 corpora x 32
    documents is ~45 min on one GPU. Spawned rather than called, so a dropped
    client does not cancel it.
    """
    band = load_config(config).band
    spec = {
        "config_name": config,
        "domains": [c.strip() for c in corpora.split(",") if c.strip()],
        "n_per": n_per,
        "layers": list(band)[::layer_stride] if layer_stride > 1 else None,
        "limit": limit or None,
        "out_dir": RESULTS,
        "corpus_cache": f"{RESULTS}/energy_corpora.json",
        "target_words": [w.strip() for w in words.split(",") if w.strip()] or None,
        "t_stride": t_stride,
        "dim_batch": dim_batch or None,
    }
    call = _fn("component_shard", gpu).spawn(spec)
    print(f"spawned {call.object_id}; waiting (safe to disconnect)")
    print(call.get())


def two_hop(
    config: str = "qwen3-8b",
    means: str = (
        f"{RESULTS}/rq1/group_means_full_qwen3-8b_dom2x32_57462165.pt,"
        f"{RESULTS}/rq1/group_means_full_qwen3-8b_dom6x32_7e3215f2.pt"
    ),
    groups: str = "gsm8k,wikitext_a",
    strengths: str = "1.0,2.0,4.0",
    layer_stride: int = 3,
    directions: str = "",
    arms: str = "",
    gpu: str = GPU,
    memory: int = 131072,
) -> None:
    """Swap a LATENT intermediate and require the second hop to follow.

    Grades three things apart: the downstream answer the swap implies
    (success), the original answer (no effect), and the swapped token showing
    up in the text (a write that reached the surface but not the computation).
    """
    band = load_config(config).band
    spec = {
        "config_name": config,
        "means_paths": [m for m in means.split(",") if m],
        "groups": [g.strip() for g in groups.split(",") if g.strip()],
        "strengths": [float(x) for x in strengths.split(",")],
        "layers": list(band)[::layer_stride] if layer_stride > 1 else None,
        "out_dir": RESULTS,
        "directions_paths": [
            f"{RESULTS}/rq1/group_mean_digests_{ns.strip()}.pt"
            for ns in directions.split(",")
            if ns.strip()
        ],
        "arms": [a.strip() for a in arms.split(",") if a.strip()],
    }
    call = _fn("two_hop_shard", gpu, memory).spawn(spec)
    print(f"spawned {call.object_id}; waiting (safe to disconnect)")
    print(call.get())


COMMANDS = {
    name: globals()[name]
    for name in [
        "warm",
        "gate",
        "gate_grid",
        "c1",
        "rq2",
        "rq1",
        "pool",
        "smoke27",
        "smoke",
        "lens",
        "c4",
        "c4_norms",
        "loading",
        "c5",
        "c7",
        "domains",
        "pool_domains",
        "domain_causal",
        "readout",
        "probe_readout",
        "lens_prior",
        "energy",
        "components",
        "two_hop",
        "geometry",
        "project_out",
        "beta_family",
        "horizon",
        "horizon_effect",
        "fluency",
        "comp_readout",
        "ablation",
        "park",
        "diffmean",
    ]
}


def main(argv: list[str] | None = None) -> None:
    """``python modal_app.py <command> [--flag value ...]``.

    The parser is generated from each launcher's signature rather than written
    out, because ``modal run`` used to generate exactly that CLI from the same
    annotations -- hand-copying fifteen signatures into argparse would be four
    hundred lines that go stale the first time a default changes. Booleans
    become ``--flag`` / ``--no-flag`` pairs; everything else is typed by its
    annotation.

    Deploy once before the first launch, and again whenever the image or a
    remote function's body changes::

        modal deploy modal_app.py
        python modal_app.py rq1 --config qwen3-8b --n-shards 8
    """
    global MAX_CONTAINERS

    import argparse
    import inspect

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, fn in COMMANDS.items():
        sub = subparsers.add_parser(name, help=(fn.__doc__ or "").split("\n")[0])
        # Global, not per-launcher: the cap lives on the shared _fn helper, so
        # every command honours it without any of them naming it in a signature.
        sub.add_argument(
            "--max-containers",
            dest="_max_containers",
            default=MAX_CONTAINERS,
            type=int,
            help=f"ceiling on simultaneous GPU containers (default {MAX_CONTAINERS})",
        )
        for param in inspect.signature(fn).parameters.values():
            flag = "--" + param.name.replace("_", "-")
            if isinstance(param.default, bool):
                sub.add_argument(
                    flag,
                    dest=param.name,
                    default=param.default,
                    action=argparse.BooleanOptionalAction,
                )
            else:
                sub.add_argument(
                    flag,
                    dest=param.name,
                    default=param.default,
                    type=type(param.default) if param.default is not None else str,
                )

    args = vars(parser.parse_args(argv))
    MAX_CONTAINERS = args.pop("_max_containers")
    print(f"cap: at most {MAX_CONTAINERS} GPU containers at once")
    COMMANDS[args.pop("command")](**args)


if __name__ == "__main__":
    main()
