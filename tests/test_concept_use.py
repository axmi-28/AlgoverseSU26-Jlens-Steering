"""Invariants for C32-C35 (jsteer.concept_use), not steering outcomes."""

import re

import pytest
import torch

from jsteer import concept_use as cu
from jsteer.steering import ResidualEdit, greedy_continuation, next_token_logits
from tests.tiny_model import TinyDecoder


@pytest.fixture(scope="module")
def hf_model():
    """A tiny random Qwen3 behind the real HFLensModel wrapper (cache path)."""
    qwen3 = pytest.importorskip("transformers.models.qwen3.modeling_qwen3")
    from jlens.hf import HFLensModel

    cfg = qwen3.Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=256,
    )
    torch.manual_seed(0)
    hf = qwen3.Qwen3ForCausalLM(cfg).float().eval()
    return HFLensModel(hf, TinyDecoder().tokenizer)


def _prefix_edit(P, d, B, layers, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {l: torch.randn(B, P, d, generator=g) * 0.5 for l in layers}


@pytest.mark.parametrize("which", ["tiny", "hf"])
def test_cached_decoding_matches_recompute(which, hf_model):
    """Prompt-only cached decoding == greedy_continuation's full recompute."""
    model = TinyDecoder(d_model=16, vocab_size=32) if which == "tiny" else hf_model
    prompt = "abcdefghij"
    ids = model.encode(prompt)
    P = 6
    deltas = _prefix_edit(P, model.d_model, 3, [1, 2])
    res = cu.evaluate(model, ids, deltas, gen_tokens=5)
    for b in range(3):
        edit = ResidualEdit({l: d[b] for l, d in deltas.items()}, positions=list(range(P)))
        ref = greedy_continuation(model, prompt, edit=edit, n_tokens=5)
        assert res["texts"][b] == ref


def test_batched_logprobs_and_kl_match_single(hf_model):
    model = hf_model
    prompt = "hello world"
    ids = model.encode(prompt)
    P = 5
    deltas = _prefix_edit(P, model.d_model, 4, [0, 3], seed=1)
    clean = torch.log_softmax(next_token_logits(model, prompt), -1)
    res = cu.evaluate(model, ids, deltas, answers={"a": [3, 7]}, ref_logprobs=clean, chunk=3)
    for b in range(4):
        edit = ResidualEdit({l: d[b] for l, d in deltas.items()}, positions=list(range(P)))
        lp = torch.log_softmax(next_token_logits(model, prompt, edit=edit), -1)
        kl = float((clean.exp() * (clean - lp)).sum())
        assert res["kl"][b] == pytest.approx(kl, rel=1e-4, abs=1e-6)
        # a two-token answer's log-prob is at most its first token's
        assert res["ans"]["a"][b] <= float(lp[3]) + 1e-5


def test_first_answer_token_logprob_is_exact(hf_model):
    model = hf_model
    ids = model.encode("xyz")
    deltas = _prefix_edit(2, model.d_model, 2, [1], seed=2)
    res = cu.evaluate(model, ids, deltas, answers={"a": [5]})
    for b in range(2):
        edit = ResidualEdit({1: deltas[1][b]}, positions=[0, 1])
        lp = torch.log_softmax(next_token_logits(model, "xyz", edit=edit), -1)
        assert res["ans"]["a"][b] == pytest.approx(float(lp[5]), abs=1e-4)


def test_prefix_edit_is_identical_across_suffixes(hf_model):
    """The paired design: prefix residuals do not depend on what follows."""
    model = hf_model
    a = model.encode("prefix text then one query")
    b = model.encode("prefix text then another, longer query here")
    P = 10
    assert torch.equal(a[0, :P], b[0, :P])
    ha = cu.band_residuals(model, a, [1, 2])
    hb = cu.band_residuals(model, b, [1, 2])
    for l in (1, 2):
        torch.testing.assert_close(ha[l][:P], hb[l][:P], rtol=1e-5, atol=1e-5)


def test_panel_units_are_well_formed():
    units = cu.build_units()
    keys = [u.key for u in units]
    assert len(keys) == len(set(keys))
    for u in units:
        kinds = [q.kind for q in u.queries]
        assert kinds[0] == "express" and kinds.count("neutral") == 2
        for q in u.queries:
            if q.kind != "neutral":
                assert cu._disjoint(q.target, q.source), (u.key, q.relation)
        if u.clue != "name":
            assert u.source.casefold() not in u.prefix.casefold()
    # every latent clue avoids naming ANY answer of its own entity
    panel = cu.load_panel()
    for ty, spec in panel["types"].items():
        for e, E in spec["entities"].items():
            for clue in E["clues"]:
                for rel in spec["relations"]:
                    for ans in E.get(rel, []):
                        pattern = r"\b" + re.escape(ans.casefold()) + r"\b"
                        assert not re.search(pattern, clue.casefold()), (e, rel, clue)


def test_entity_split_is_disjoint_and_halved():
    split = cu.entity_split(cu.load_panel())
    units = cu.build_units()
    for u in units:
        if u.split != "cross":
            assert split[u.source] == split[u.target] == u.split
    assert {"dev", "test"} == set(split.values())


def test_swap_deltas_match_norms_and_full_is_the_reference():
    class FakeBank:
        d = 12

        def __init__(self):
            g = torch.Generator().manual_seed(3)
            self.r = {}
            self.g = g
            self.u = {"A": torch.randn(12, generator=g), "B": torch.randn(12, generator=g)}

        def vec(self, method, word, layer):
            key = (method, word, layer)
            if key not in self.r:
                self.r[key] = torch.randn(12, generator=self.g)
            return self.r[key]

    bank = FakeBank()
    unit = cu.Unit("t", "A", "B", "0", "p", "dev", ())
    h = {5: torch.randn(9, 12)}
    arms = list(cu.SWAP_WRITERS)
    out = cu.swap_deltas(bank, h, unit, "c", arms, P=7)
    ref = out["full"][5]
    assert ref.shape == (7, 12)
    for arm in arms:
        if cu.SWAP_WRITERS[arm][1]:
            torch.testing.assert_close(out[arm][5].norm(dim=-1), ref.norm(dim=-1))


def test_position_sets_exclude_the_sink():
    assert cu.position_index("prefix", 6, 10) == [1, 2, 3, 4, 5]
    assert cu.position_index("last", 6, 10) == [5]
    assert cu.position_index("tail3", 6, 10) == [3, 4, 5]
    assert cu.position_index("prompt", 6, 10) == list(range(1, 10))


def test_eligibility_requires_express():
    rows = [
        {"key": "k", "relation": "express", "kind": "express", "filler": 0, "hit_source": True, "hit_target": False},
        {"key": "k", "relation": "capital", "kind": "use", "filler": 0, "hit_source": True, "hit_target": False},
        {"key": "k", "relation": "language", "kind": "use", "filler": 0, "hit_source": False, "hit_target": False},
        {"key": "k", "relation": "express", "kind": "express", "filler": 1, "hit_source": False, "hit_target": False},
        {"key": "k", "relation": "capital", "kind": "use", "filler": 1, "hit_source": True, "hit_target": False},
    ]
    assert cu.freeze_eligibility(rows) == {"k": {"0": ["capital", "express"]}}


def test_manifest_changes_with_every_scientific_input(tmp_path):
    a = cu.ConceptSpec(stage="swap", swap_arms=["full"], out_dir=str(tmp_path))
    b = cu.ConceptSpec(stage="swap", swap_arms=["full"], doses=[1.0], out_dir=str(tmp_path))
    c = cu.ConceptSpec(stage="swap", swap_arms=["full"], shard=1, n_shards=2, out_dir=str(tmp_path))
    assert a.tag != b.tag
    assert a.tag == c.tag  # sharding is not a scientific input
    assert a.out_path() != c.out_path()


def test_probe_returns_first_token_logprobs(hf_model):
    model = hf_model
    ids = model.encode("probe me")
    deltas = _prefix_edit(3, model.d_model, 2, [2], seed=7)
    res = cu.evaluate(model, ids, deltas, probe=[4, 9])
    for b in range(2):
        edit = ResidualEdit({2: deltas[2][b]}, positions=[0, 1, 2])
        lp = torch.log_softmax(next_token_logits(model, "probe me", edit=edit), -1)
        assert res["probe"][b] == pytest.approx([float(lp[4]), float(lp[9])], abs=1e-4)


def test_a_zero_row_stays_zero_and_is_counted():
    """An arm whose two directions coincide has nothing to write at a position."""
    ref = torch.tensor([[3.0, 4.0], [1.0, 0.0]])
    delta = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    report = {}
    out = cu._match_or_keep(delta, ref, report, "arm/L1")
    torch.testing.assert_close(out[0].norm(), ref[0].norm())
    assert out[1].norm() == 0
    assert report == {"arm/L1": 1}
    with pytest.raises(ValueError, match="writes nothing"):
        cu._match_or_keep(torch.zeros(2, 2), ref, {}, "arm/L1")


class _ScaleBank:
    """Bank with controllable per-word, per-family norms."""

    d = 12

    def __init__(self):
        g = torch.Generator().manual_seed(11)
        self.g = g
        self.r = {}
        self.u = {w: torch.randn(12, generator=g) for w in ("A", "B")}

    def vec(self, method, word, layer):
        key = (method, word, layer)
        if key not in self.r:
            v = torch.randn(self.d, generator=self.g)
            scale = {"c_full": 3.0, "c_off": 1.0}.get(method, 1.0)
            self.r[key] = v / v.norm() * scale * (2.0 if word == "B" else 1.0)
        return self.r[key]


def test_legacy_swap_arms_keep_their_raw_columns():
    bank = _ScaleBank()
    unit_ = cu.Unit("t", "A", "B", "0", "p", "dev", ())
    V = cu.arm_columns(bank, unit_, "c", "off_m", 5)
    torch.testing.assert_close(V[:, 0], bank.vec("c_off", "A", 5))
    torch.testing.assert_close(V[:, 1], bank.vec("c_off", "B", 5))
    assert cu.arm_spec("off_m") == {
        "fam": "off", "matched": True, "cn": False, "kappa": None, "kappa_from": None
    }


def test_column_normalised_arms_have_kappa_one():
    bank = _ScaleBank()
    unit_ = cu.Unit("t", "A", "B", "0", "p", "dev", ())
    for arm in ("full_cn", "off_cn"):
        V = cu.arm_columns(bank, unit_, "c", arm, 5)
        assert V[:, 0].norm() == pytest.approx(1.0, abs=1e-5)
        assert V[:, 1].norm() == pytest.approx(1.0, abs=1e-5)


def test_explicit_and_donated_kappa():
    bank = _ScaleBank()
    unit_ = cu.Unit("t", "A", "B", "0", "p", "dev", ())
    V = cu.arm_columns(bank, unit_, "c", "full_k2", 5)
    assert (V[:, 1].norm() / V[:, 0].norm()).item() == pytest.approx(2.0, abs=1e-5)
    donated = cu.arm_columns(bank, unit_, "c", "full_koff", 5)
    off = cu.arm_columns(bank, unit_, "c", "off_m", 5)
    assert (donated[:, 1].norm() / donated[:, 0].norm()).item() == pytest.approx(
        (off[:, 1].norm() / off[:, 0].norm()).item(), abs=1e-5
    )


def test_clamp_is_invariant_to_uniform_but_not_relative_rescaling():
    """Why kappa is the free parameter that norm matching cannot fix."""
    g = torch.Generator().manual_seed(4)
    h = torch.randn(6, 12, generator=g)
    V = torch.randn(12, 2, generator=g)

    def delta(M):
        c = cu.coordinates(h, M)
        return (c.flip(-1) - c) @ M.T

    torch.testing.assert_close(delta(V), delta(V * 7.0), rtol=1e-4, atol=1e-4)
    skewed = V.clone()
    skewed[:, 1] *= 3.0
    assert not torch.allclose(delta(V), delta(skewed), rtol=1e-2, atol=1e-2)


def test_lambda_family_interpolates_off_and_diag_exactly():
    class Bank2:
        d = 8

        def __init__(self):
            g = torch.Generator().manual_seed(21)
            self.rows = {("c_off", 3): torch.randn(8, generator=g),
                         ("c_diag", 3): torch.randn(8, generator=g)}
            self.index = {}
            self.u = {"A": torch.randn(8, generator=g)}

        vec = cu.Bank.vec

        def __getattr__(self, name):
            raise AttributeError(name)

    b = Bank2()
    b.rows = {("c_off", 3): {0: b.rows[("c_off", 3)]}, ("c_diag", 3): {0: b.rows[("c_diag", 3)]}}
    # exercise the arithmetic directly instead of the row store
    off = torch.randn(8, generator=torch.Generator().manual_seed(1))
    diag = torch.randn(8, generator=torch.Generator().manual_seed(2))

    class Simple:
        d = 8
        u = {"A": torch.ones(8)}

        def vec(self, method, word, layer):
            if "_lam" in method:
                return cu.Bank.vec(self, method, word, layer)
            return {"c_off": off, "c_diag": diag, "c_full": off + diag}[method]

    s = Simple()
    torch.testing.assert_close(s.vec("c_lam0.0", "A", 1), off)
    torch.testing.assert_close(s.vec("c_lam1.0", "A", 1), off + diag)
    torch.testing.assert_close(s.vec("c_lam1.0", "A", 1), s.vec("c_full", "A", 1))
    torch.testing.assert_close(s.vec("c_lam0.5", "A", 1), off + 0.5 * diag)


def test_lambda_arm_names_parse():
    assert cu.arm_spec("lam0.5_cn") == {
        "fam": "lam0.5", "matched": True, "cn": True, "kappa": None, "kappa_from": None
    }
    assert cu.arm_spec("lam0.5_m")["cn"] is False
    assert cu.arm_spec("lam0.5_m")["matched"] is True


def test_unit_column_swap_is_a_householder_reflection():
    """C38: for unit s, t the implemented swap is h -> h - 2w(w^T h), w = unit(t - s)."""
    g = torch.Generator().manual_seed(3)
    d = 24
    s, t = cu.unit(torch.randn(d, generator=g)), cu.unit(torch.randn(d, generator=g))
    h = torch.randn(9, d, generator=g) * 5
    V = torch.stack([s, t], 1)
    geo = cu.householder_sites(h, V)
    w = geo["w"]
    c = cu.coordinates(h, V)
    delta = (c.flip(-1) - c) @ V.T
    torch.testing.assert_close(delta, -2 * (h @ w)[:, None] * w[None], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(geo["native"], geo["m_clean"], rtol=1e-4, atol=1e-4)
    assert abs(geo["rho"] - float(s @ t)) < 1e-6
    # the operator depends on the line, not the sign: swapping s and t flips w only
    geo2 = cu.householder_sites(h, torch.stack([t, s], 1))
    torch.testing.assert_close(geo2["m_clean"], geo["m_clean"])


def test_measured_edit_records_the_state_before_its_own_edit(hf_model):
    """h~ at the first edited layer is the clean state; later layers see earlier edits."""
    model = hf_model
    ids = model.encode("abcdefghijkl")
    layers = [1, 2]
    P, B = ids.shape[1], 3
    deltas = _prefix_edit(P, model.d_model, B, layers)
    g = torch.Generator().manual_seed(1)
    axes = {l: torch.stack([cu.unit(torch.randn(model.d_model, generator=g)) for _ in range(B)]) for l in layers}
    h0 = cu.band_residuals(model, ids, layers)
    _, stats = cu._teacher_forced(model, ids, [5], deltas, axes)
    for b in range(B):
        torch.testing.assert_close(stats[1][0][b], 2 * (h0[1] @ axes[1][b]).abs(), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(stats[1][1][b], h0[1].norm(dim=-1), rtol=1e-4, atol=1e-4)
    # layer 2 receives layer 1's edit, so it must differ from the clean state
    assert not torch.allclose(stats[2][1][0], h0[2].norm(dim=-1), rtol=1e-3, atol=1e-3)
    # and the measuring hook must still apply the edit: log-probs match a plain edit
    lp_plain, _ = cu._teacher_forced(model, ids, [5], deltas)
    lp_meas, _ = cu._teacher_forced(model, ids, [5], deltas, axes)
    torch.testing.assert_close(lp_plain, lp_meas)


def test_dynamic_householder_hits_target_norm_and_reduces_to_reflection(hf_model):
    """C38b: norm exactly m*, along +/-w; with m* = 2|w^T h~| it is the reflection itself."""
    model = hf_model
    ids = model.encode("abcdefghijkl")
    T, B, layers = ids.shape[1], 2, [1, 2]
    g = torch.Generator().manual_seed(4)
    axes = {l: torch.stack([cu.unit(torch.randn(model.d_model, generator=g)) for _ in range(B)]) for l in layers}
    mstar = {l: torch.rand(B, T, generator=g) + 0.5 for l in layers}
    _, edit = cu._tf_dynamic(model, ids, [5], axes, mstar)
    for l in layers:
        live = ~edit.skipped[l]
        torch.testing.assert_close(edit.achieved[l][live], mstar[l][live], rtol=1e-5, atol=1e-5)
    # single layer, m* = the natural magnitude on the clean state: identical to a Householder delta
    h0 = cu.band_residuals(model, ids, [1])[1]
    w = axes[1]
    nat = {1: 2 * torch.einsum("pd,bd->bp", h0, w).abs()}
    lp_dyn, _ = cu._tf_dynamic(model, ids, [5], {1: w}, nat)
    hh = {1: torch.stack([-2 * (h0 @ w[b])[:, None] * w[b][None] for b in range(B)])}
    lp_ref, _ = cu._teacher_forced(model, ids, [5], hh)
    torch.testing.assert_close(lp_dyn, lp_ref, rtol=1e-4, atol=1e-4)


def test_clean_site_grads_match_finite_differences(hf_model):
    """C39: summed g_a . delta_a over several sites = the multi-site directional derivative."""
    import copy

    model = copy.deepcopy(hf_model)
    for mod in {m for m in (getattr(model, k, None) for k in vars(model)) if isinstance(m, torch.nn.Module)}:
        mod.double()
    ids = model.encode("abcdefghijkl")
    T, layers, a_ids = ids.shape[1], [1, 2], [5, 7]
    g = cu.clean_site_grads(model, ids, a_ids, layers, T)
    deltas = {l: d.double() for l, d in _prefix_edit(T, model.d_model, 1, layers, seed=9).items()}
    base, _ = cu._teacher_forced(model, ids, a_ids, {})
    eps = 1e-3  # log-probs are float32 in _teacher_forced; smaller steps are rounding
    up, _ = cu._teacher_forced(model, ids, a_ids, {l: d * eps for l, d in deltas.items()})
    dn, _ = cu._teacher_forced(model, ids, a_ids, {l: -d * eps for l, d in deltas.items()})
    fd = float(up[0] - dn[0]) / (2 * eps)
    pred = sum(float((g[l] * deltas[l][0]).sum()) for l in layers)
    assert abs(fd - pred) < 5e-3 * max(1.0, abs(pred))
    # dropping the layer-1 -> layer-2 path (a wiring bug) must be detectable
    g1_only = float((g[1] * deltas[1][0]).sum())
    assert abs(fd - g1_only) > 1e-2
    # the gradient pass must not leave the model's parameters trainable or hooked
    assert all(len(m._forward_hooks) == 0 for m in model.layers)
