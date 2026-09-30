# Reproducing the experiments

Run commands from the repository root after the README's installation steps.
The supported installation is an editable source checkout: configs and evaluation
definitions live alongside `src/`. A standalone wheel is not a self-contained
experiment distribution.

## 1. Recompute from recorded runs (CPU)

```bash
python scripts/release.py verify
python scripts/release.py restore
python scripts/reproduce.py tables
python scripts/reproduce.py latent
python scripts/reproduce.py gradients
python scripts/reproduce.py magnitude
python scripts/reproduce.py figures
# Or run all of the above analyses:
python scripts/reproduce.py all
```

`verify` checks compressed and uncompressed SHA-256 values for all 49 reference
artifacts. `restore` materializes them into `results/` and the upstream prompt
cache; it refuses to overwrite differing files. The analysis entrypoint checks
all restored input hashes before running. No model, network or cloud service is
needed once the Python environment has been installed.

| Command | Output |
|---|---|
| `tables` | `output/reproduced/tables.json` and `tables.md`; latent, replicate, interpolation and direction-scale values with input hashes |
| `latent` | `output/reproduced/latent.md`; frozen-panel verification, counts, paired differences and category-bootstrap intervals |
| `gradients` | `results/c39_gradgeom_qwen3-8b.md` and diagnostic plots; includes float32 finite-difference gate |
| `magnitude` | `results/c38_magnitude_qwen3-8b.md`, configuration and plots; includes spline adjustment and runtime-reflection control |
| `figures` | `output/figures/final/` and `output/pdf/jlens_final_visual_review.pdf`; source audit and vector figure assets |

The figures are the final available *scripted* figure suite. The manuscript's
separate `fig_overview.png` has no identified source generator in this checkout;
see [the audit](PAPER_AUDIT.md). Plot fonts can differ across systems (Arial falls
back to DejaVu Sans). Numerical values, rather than PDF byte hashes, are the
appropriate cross-platform check.

The compressed bundle is about 132 MiB; restoration requires about 272 MiB.
Leave additional space for plots, an environment and temporary analysis arrays.
Magnitude bootstrapping is the slowest CPU analysis. Run it separately if needed.

## 2. Rerun the principal experiments (GPU)

The original remote launcher defaulted to **A100-80GB** for Qwen3-8B. Other GPUs
may work with smaller cotangent batches, but have not been validated here.
Base-model weights and the downloaded reference lens are fetched from Hugging
Face. This release does not run any paid cloud job automatically.

`run_paper.py` prints a plan by default. Add `--execute` on your GPU machine.
It records package versions, model configuration, source hashes and input
hashes, and writes to `results/rerun/`, separately from the historical results.
A changed run requires a new output directory.

```bash
# Refit the 70-token component bank using the frozen fitting-text snapshot.
python scripts/run_paper.py components
python scripts/run_paper.py components --execute

# Re-evaluate the frozen 55-item latent development panel, using the bundled
# directions by default. --directions PATH selects a newly fitted digest.
python scripts/run_paper.py latent --corpus gsm8k --execute
python scripts/run_paper.py latent --corpus wikitext_a --execute

# Concept interpolation: each preset has four shards (0, 1, 2, 3).
python scripts/run_paper.py c37lambdaprompt --shard 0
python scripts/run_paper.py c37lambdaprompt --shard 0 --execute
# Repeat for shards 1..3, then c37lambdaprefix and c36kappa.

# Magnitude/gradient stages: two shards (0, 1) per convention.
python scripts/run_paper.py c38magprompt --shard 0 --execute
python scripts/run_paper.py c39gradprompt --shard 0 --execute
# Repeat shard 1 and the corresponding ...prefix presets.

# Float32 validation: one shard per convention, fixed 40-unit subsample.
python scripts/run_paper.py c39checkprompt --execute
python scripts/run_paper.py c39checkprefix --execute
```

A new component fit uses `configs/paper_components.json`: 32 documents per
corpus, 128-token windows, target stride **4**, and the seven listed layers.
Use the digest path printed by the fit as `--directions PATH` for downstream
runs. The default reference directions and eligibility deliberately isolate
intervention replication from refitting variation.

To regenerate eligibility locally:

```bash
python scripts/run_paper.py clean --shard 0 --execute
python scripts/run_paper.py clean --shard 1 --execute
python scripts/run_paper.py freeze --execute
```

Then pass `--eligibility results/rerun/concept/eligibility.json` to downstream
stages. Freezing requires both complete clean shards with identical manifests
and records their checksums. Eligibility is defined by clean behavior, never
by a steered outcome. New eligibility may change denominators and should be
reported as a new run.

The dynamic `c38matchedprompt`/`c38matchedprefix` controls recompute natural
magnitudes at runtime. They are diagnostic interventions,
not a valid substitute for the static matched-norm control (see PROTOCOL.md).

## 3. Corpus experiments and optional remote execution

The original CLI supports the replicate and expanded corpus panels:

```bash
python scripts/26_tier0_replicates.py --stage fit --panel replicates --out-dir results/new-corpora
python scripts/26_tier0_replicates.py --stage causal --panel replicates --out-dir results/new-corpora
python scripts/26_tier0_replicates.py --stage report --panel replicates --results results/new-corpora

python scripts/26_tier0_replicates.py --stage fit --panel aqua --label aqua --out-dir results/new-corpora
python scripts/26_tier0_replicates.py --stage causal --panel aqua --label aqua --out-dir results/new-corpora
python scripts/26_tier0_replicates.py --stage report --panel aqua --label aqua --results results/new-corpora
```

`--panel expanded --label expanded` runs the multi-corpus comparison, and
`--panel ablations --label ablations` runs the structural corpus controls.
Corpus streams are deterministic in the code's order, but historical dataset
revisions and all replicate text snapshots were not saved. These commands
therefore constitute methodological reruns; the bundled trial outputs support
exact reanalysis of the original counts. The held-out ordered-scale panel is
`data/heldout/ordered-scale.json`; `CausalSpec(trial_set="ordered-scale")` selects
it. `EnergySpec`/`run_energy_map` implement the position-pair energy analysis.

Remote execution is optional:

```bash
python -m pip install -e '.[remote]'
modal deploy modal_app.py
python modal_app.py --help
python scripts/48_concept_use.py plan c37lambdaprompt
```

Deploy in your own account; copy restored inputs to your own `jsteer-results`
volume before launching stages that depend on them. The local GPU runner avoids
this account/volume setup entirely. Cloud launches are explicit and incur charges.
Do not copy credentials or the private working tree into an image. The image
exclusions omit drafts, notes, generated outputs and reference bundles.

## 4. Environment and numerical limits

`jlens` is pinned to commit `581d398613e5602a5af361e1c34d3a92ea82ba8e`.
`requirements/analysis-tested.txt` records the installed local environment used
for this release's tests and analyses. It is not a recovered lockfile for the
original GPU runs, which used package version ranges. The model/ tokenizer and
reference-lens revisions now come from the local Hub cache. Original remote
run records did not establish these revisions or the CUDA/driver version.

Do not promise bitwise GPU reproduction: dtype, kernels, batching and hardware
can change near-tied generations. Preserve complete manifests, source hashes,
fitting text and eligibility for any new run. The historical source hashes are
kept in recorded outputs; cleanup does not rewrite them to match the release.

Default tests are offline: `python -m pytest -q`. For optional hosted parity/
Hub checks, explicitly select `python -m pytest -m network`; these depend on
external availability. No full GPU sweep was rerun during repository cleanup.
