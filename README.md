# jsteer — when are J-Lens read directions valid write directions?

Scaffolding for the first research subcomponent: **characterize how much the
prompt-local Jacobian `J_x` varies around the averaged `J̄` the lens actually
uses, and whether that variation predicts whether steering works.**

Background and motivation live in [`research_proposal.md`](research_proposal.md)
and [`implementation.md`](implementation.md) (working drafts from Google Docs —
this README does not duplicate them, and nothing here edits them).

## What this repo does today

1. Loads a Qwen3 (or Gemma-2) model and its pre-fitted lens from
   [`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens).
2. Computes `J_x` for a prompt and source layer through the `jlens` code path —
   verified **bit-for-bit** against `jlens.fitting.jacobian_for_prompt`.
3. Computes `g_x = J_xᵀ u_y`, the pullback of an unembedding row, **batching the
   cotangents** so K target tokens cost `ceil(K / dim_batch)` backward passes
   rather than K.
4. Runs the two sanity checks the science depends on (below).

Both sanity checks have been run and pass; each script prints its measured
numbers and writes them to `results/`.

## Setup

```bash
uv venv --python 3.12
uv pip install -e ".[dev,analysis]"
.venv/bin/python -m pytest -q          # 21 tests, no model download needed
```

`google/gemma-2-2b` is a **gated** HF repo — accept the license on the model
page and `huggingface-cli login` before the parity check can run. The Qwen3
models and every lens `.pt` are ungated.

## Why the pullback is batched

`jlens.fitting.jacobian_for_prompt` materializes `J_x` by injecting `d_model`
one-hot cotangents. On Qwen3-8B that is 32 backward passes and **67 MB per
layer**. But nearly every question we want to ask is "where does the unembedding
row for token `y` pull back to?", which needs one column, not the matrix.
`jsteer.jacobian.pullback_for_prompt` takes arbitrary cotangents, so ~100 target
tokens is **one backward pass and 1.6 MB**. Measured on Qwen3-1.7B:

```
cost: full J_x = 256 passes, 16.8 MB/layer | K=4 pullbacks = 1 pass, 0.03 MB
```

The identity that licenses this — `pullback(c) == J_xᵀ c` — is asserted in
`tests/test_jacobian.py`, along with `dim_batch`-invariance, linearity, and
causality. `jacobian_for_prompt` is the identity-cotangent special case of the
same primitive, so the cheap path and the expensive path cannot drift apart.

## Layout

```
configs/            one YAML per model: dims, lens filename, workspace band,
                    and the fit settings transcribed from the lens config.yaml
src/jsteer/
  config.py         ModelConfig / FitSpec
  loading.py        model + lens loading, unembedding rows
  jacobian.py       pullback_for_prompt, jacobian_for_prompt, position sets
  metrics.py        J_x vs J̄ geometry; identity_distance
  corpus.py         reconstruction of the wikitext fitting distribution
  neuronpedia.py    client for the hosted lens API
  data.py           the paper's prompt sets (192 swap trials, 90 two-hop items)
scripts/
  01_check_lens_parity.py       sanity check 1
  02_check_fit_convergence.py   sanity check 2
  03_pullback_demo.py           g_x at scale on one trial
results/            script output (gitignored except .gitkeep)
```

## The two sanity checks

```bash
.venv/bin/python scripts/01_check_lens_parity.py --config gemma-2-2b
.venv/bin/python scripts/02_check_fit_convergence.py --config qwen3-1.7b \
    --n-prompts 3 --source-layers 26 --dim-batch 64
```

**1 — Does our `J̄` reproduce the deployed lens?** Only some models have a live
lens server; `qwen3-1.7b/4b/8b` return `"No server host found"`, so this runs on
gemma-2-2b and the Qwen3 pipeline inherits it (same code path, same lens files).
The check is *not* top-1 string equality — Neuronpedia applies a display-layer
vocabulary filter — but the rank of their top-1 inside our full ranking, with
the plain logit lens as a control. **Currently PASSes** in the workspace band.

**2 — Does averaging our own `J_x` converge to the downloaded `J̄`?** Rebuilds
the wikitext-103 fitting stream and compares both the per-prompt trace against
the lens's published `*_convergence.csv` and the running mean against `J̄`.
**Currently converging**, with the window shape (`seq_len=128`, `n_valid=111`)
matching the published trace exactly.

## Things that will bite you

- **Lens filenames follow the HF model id, not the directory name.**
  `qwen3-8b/.../Qwen3-8B_jacobian_lens.pt` but
  `gemma-2-2b/.../gemma-2-2b_jacobian_lens.pt`. There is no rule; it is written
  down per model in `configs/`.
- **The published lenses are not 1000-prompt fits.** They early-stopped at
  `prompts_fitted` ≈ 450–480 (466 for qwen3-1.7b), on wikitext-103, not a
  general web corpus.
- **`skip_first=16` makes the fitting convention unusable on the eval prompts.**
  "The capital of France is the city of" is 8 tokens. Position sets are
  therefore an explicit parameter (`source_positions` / `target_positions`), not
  an inherited default — see the note in `jsteer.jacobian.resolve_positions`.
- **Early layers are a different regime.** On gemma-2-2b the J-lens readout
  matches the hosted lens from L10 up but diverges at L1-L9, where `J̄` is far
  from the identity (`identity_distance` 8.75 at L0 vs 0.52 at L26 on
  qwen3-1.7b). Parity is therefore scored inside the workspace band, with
  out-of-band numbers printed as diagnostics. The per-model bands in
  `configs/*.yaml` are measured for gemma-2-2b and transposed for the Qwen3
  models -- check them directly before relying on them.
