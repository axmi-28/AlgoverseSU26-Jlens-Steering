# Broader model replication

The [combined core-results table](results/index.html) and
[Markdown version](results/full_table.md) include the original
Qwen references, both replication campaigns, initial Gemma/Llama runs, and
separately labeled exploratory/projection controls. Rebuild locally with
`python scripts/75_all_core_results.py`; this makes no model or GPU calls.

Cross-model figures summarize the strength-1 effects for
[published Neuronpedia lenses](results/figures/neuronpedia_across_models.png)
and [self-fitted lenses](results/figures/self_fitted_across_models.png),
showing latent-swap success and leakage on the same trials, with per-model
trial counts, matching axes and paired 95% intervals. SVG exports, plotting data and
[captions](results/figures/captions.md) are in the same directory.
Rebuild with `python scripts/76_core_results_figures.py`.

## Source policy for future experiments

The protocol adopted on 2026-10-18 uses Neuronpedia's published J-lenses
for future runs. The completed self-fitted campaign is preserved separately. Do not silently fit a replacement when a
published lens is missing. See `data/protocols/lens_source_policy.json`.

`data/protocols/neuronpedia_gemma_llama_inventory.json` records six available
Gemma/Llama checkpoint/lens pairs, with pinned model and lens revisions. Their
separate `-np` configurations preserve the configurations used by past runs.

The released full-lens checkpoint does not provide separate same-position and
off-diagonal components. A future protocol must explicitly distinguish a
published lens minus an estimated diagonal contribution from an unembedding
projection proxy. Neither may be described as an exact decomposition of the
released lens without additional evidence. The published-lens campaign uses the released
full lens plus an estimated same-position contribution. The frozen protocol is
`data/protocols/neuronpedia_replication.json`: estimate only the diagonal on
64 frozen WikiText chunks, retain the released full lens unchanged, and
subtract the estimate. The first 32 chunks provide a fixed sensitivity check;
split-half cosine diagnostics quantify diagonal-estimate stability. This does
not reconstruct the original published fitting run.

The six models are Gemma 2 2B/9B IT, Gemma 3 4B IT/12B PT, and Llama 3.1 8B
base/instruct. The same persistent app and two-H100 cap apply. Gated weights
use an ephemeral credential, never a credential written to results or volumes.
No fresh full lens is fitted. The baseline is the published lens directly;
the filtered direction is labeled as estimated throughout the reports.

```bash
.venv/bin/modal deploy modal_replication.py
.venv/bin/python scripts/72_neuronpedia_replication.py plan
.venv/bin/python scripts/72_neuronpedia_replication.py start
.venv/bin/python scripts/72_neuronpedia_replication.py status
.venv/bin/python scripts/72_neuronpedia_replication.py fetch
.venv/bin/python scripts/73_neuronpedia_report.py results/neuronpedia_replication --figure
.venv/bin/python scripts/69_filtering_replication_sensitivity.py results/neuronpedia_replication
.venv/bin/python scripts/74_neuronpedia_audit.py results/neuronpedia_replication --remote
```

This campaign writes `jsteer-results:/neuronpedia_replication` and
`results/neuronpedia_replication`. Its source policy, model/lens revisions,
evaluation panels, fitting texts and scientific dependencies are hashed before
launch. Model-specific clean eligibility and a new fitting corpus prevent
attributing differences from the historical campaign solely to lens source.
The optional currency analysis reuses the earlier, frozen alias audit and is
exploratory. The artifact audit verifies complete paired keys, published-file
checksums, unchanged full directions, both diagonal estimates, edit norms and
exactly six unique successful GPU invocations across supervisor recovery.

The commands below document the completed, self-fitted campaign and are for
reproduction of that historical protocol, not new experiments under this policy.

## Completed self-fitted campaign

The frozen design is `data/protocols/filtering_replication.json`. It extends
`cross_model_filtering.json`, used for the Gemma-3-12B and Llama-3.1-8B runs.
This extension uses Qwen3-8B as a fresh positive control, Qwen3-4B and 14B for
scale, Qwen2.5-7B-Instruct for generation, and Mistral-7B-Instruct-v0.3 and
OLMo-2-1124-7B-Instruct for additional families. All checkpoint revisions are
pinned in `configs/`. Official checkpoint sources:

- [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B),
  [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B),
  [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B)
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- [OLMo-2-1124-7B-Instruct](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct)

One deployed app, `jlens-filtering-replication`, owns preparation, a durable
CPU supervisor and a GPU function with a deployment-level maximum of two
containers. Launch through deployed function handles; do not use `modal run`.
Redeployments update this app instead of creating ephemeral app records.
Other workspace applications are not stopped or modified.

Each model has a one-hour GPU invocation limit, zero retries and a 3500-second
internal scientific budget. Six invocations therefore have at most six billed
GPU invocation hours in total; image builds and CPU preparation are separate.
Models failing eligibility or validation are recorded as failures, never
silently replaced. A recovered supervisor reattaches to recorded GPU call IDs;
it does not rerun completed or pending model invocations. An existing local or remote launch ledger prevents a
duplicate campaign. The supervisor continues independently of the local shell.

```bash
.venv/bin/modal deploy modal_replication.py
.venv/bin/python scripts/66_filtering_replication.py plan
.venv/bin/python scripts/66_filtering_replication.py start
.venv/bin/python scripts/66_filtering_replication.py status
.venv/bin/python scripts/66_filtering_replication.py fetch
```

Results live in `jsteer-results:/filtering_replication` and locally in
`results/filtering_replication`. The supervisor writes `report.md`,
`summary.md`, `summary.json` and `campaign.json` when runs finish. Rebuild the
reports locally after a fetch with:

```bash
.venv/bin/python scripts/60_cross_model_report.py results/filtering_replication
.venv/bin/python scripts/67_filtering_replication_report.py results/filtering_replication
.venv/bin/python scripts/68_filtering_replication_figure.py results/filtering_replication
.venv/bin/python scripts/69_filtering_replication_sensitivity.py results/filtering_replication
.venv/bin/python scripts/70_filtering_replication_audit.py results/filtering_replication
```

The runner hashes the checkpoint configuration, frozen panels, protocol and
scientific dependencies and rejects a mismatch before loading a model. Every
run records dependency versions and GPU identity, native-forward and independent
pullback checks, all 32 fitting texts, the full/diagonal/off decomposition,
per-site edit-norm matching and complete paired result keys.

Accuracy and incorrect name emission are separate endpoints. Paired source
cluster intervals are conditional on one fitted lens and are pointwise, without
multiple-comparison correction. Interpret the task and fixed strength together;
do not choose a strength after seeing outcomes. Model-specific clean eligibility
changes the evaluation population. This campaign tests two core tasks with one
GSM8K fit per model; it does not establish replication of the entire paper,
cross-corpus robustness or fit-seed stability.

The comparison figure exports PNG, SVG and its source statistics under
`output/filtering_replication`. Earlier Gemma/Llama results are labeled as
historical. Its three panels retain both fixed latent strengths and the
concept-use contrast, including negative or inconclusive outcomes.

`currency_sensitivity.md` and `population_audit.md` are explicitly exploratory
CPU follow-ups, separate from the frozen primary comparisons. They audit full
currency-name aliases, exclude all currency questions, stratify concept use by
entity type and compare exactly shared queries with Qwen3-8B. The alias plan
records that the first Qwen/Mistral outputs had already been inspected.
