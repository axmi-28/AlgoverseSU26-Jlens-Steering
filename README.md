# Investigating J-lens steering with position-pair filtering

This repo hosts the research code for the paper Investigating J-Lens Steering with Position-Pair Filtering, on testing how the construction of a Jacobian lens changes the
causal effects of writing its directions into a language model. The principal
experiments are on Qwen3-8B.

## Start here

- [Reproduce the results](docs/REPRODUCING.md): offline verification, reports,
  figures, and GPU reruns.
- [Model card](MODEL_CARD.md): model, fitting data, interventions, intended use,
  evaluation, and limitations.
- [Paper-to-code map](docs/EXPERIMENTS.md): which experiment supports each result.
- [Paper audit](docs/PAPER_AUDIT.md): known discrepancies that require manuscript
  review before claiming exact reproduction of every printed number.
- [Protocol](docs/PROTOCOL.md) and [data provenance](data/README.md).

## Install and verify

Use Python 3.12 from the repository root. The exact environment used for the
release's CPU checks is recorded in `requirements/analysis-tested.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/analysis-tested.txt
python -m pip install --no-deps -e .
python -m pytest -q
python scripts/release.py verify
python scripts/release.py restore
python scripts/reproduce.py tables
```

The default tests exclude network/model-download tests. The bundled reference
artifacts are losslessly compressed raw outputs and required fitted directions,
with checksums. Restoring them requires about 272 MiB. It does not download a
model, run a GPU, or contact a cloud account. Differing existing results are
never overwritten.

For a flexible dependency install instead of the tested pins, use
`python -m pip install -e '.[dev,analysis]'`. Numerical agreement with the tested
environment must then be checked. See the reproduction guide for GPU hardware
and dependency provenance limits.

## Repository layout

| Path | Contents |
|---|---|
| `src/jsteer/` | Jacobians, position-pair decomposition, steering, grading, corpora and experiment runners |
| `configs/` | Model dimensions, loading revisions and workspace bands |
| `data/heldout/` | Frozen evaluation definitions |
| `data/protocols/` | Frozen latent-panel design |
| `data/reference/` | Checksummed, compressed inputs and trial-level outputs used by the paper |
| `scripts/reproduce.py` | CPU reproduction entrypoint |
| `scripts/run_paper.py` | Local experiment plans and GPU execution |
| `scripts/` | Original numbered experiment/report entrypoints; [catalog](docs/SCRIPTS.md) |
| `tests/` | Small-model and mathematical correctness checks |
| `modal_app.py` | Optional remote execution; requires your own Modal account |
| `results/`, `output/` | Generated files; ignored by Git |

The numbered scripts retain their names because run provenance and analysis
imports refer to them. Older exploratory experiments are identified in the
catalog and are not evidence for cross-model claims in this paper.

## Anonymous distribution

```bash
python scripts/release.py audit
python scripts/release.py archive
```

This creates `dist/jlens-steering-anonymous.zip` from an explicit inclusion
policy, with fixed archive timestamps and file checksums. It excludes Git
history, drafts, manuscripts, logs, credentials and local working notes. A
public repository's owner, URL and existing history remain identifying; use
the standalone archive or a separately hosted anonymous copy for review.
See [release notes](docs/RELEASE.md).

Original code: [MIT License](LICENSE). Third-party data, models and dependencies
retain their [upstream terms](THIRD_PARTY_NOTICES.md).
