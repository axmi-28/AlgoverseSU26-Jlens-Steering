# Cross-model filtering figures

The manuscript figure `cross_model_deduplicated.pdf` / `.svg` / `.png` combines
seven self-fitted and four published-lens conditions. The separate
`self_fitted_across_models` and `neuronpedia_across_models` files show the same
disjoint selections. Vector PDF exports can be included directly in Overleaf.

## Display selection

One row per model/size: prefer instruction-tuned over base variants, published
lenses over self-fitted lenses for the same checkpoint, and the newer Qwen3-8B
replication over its original GSM8K/WikiText references. The rule does not use
accuracy or leakage outcomes, and no conditions are averaged together.
The full source table retains all 16 conditions on 13 checkpoints; the plot
shows 11 distinct checkpoints. Its selection must not be described as all 16
conditions or as a controlled comparison of lens sources.

Omitted from the plot only: P-GSM and P-WIKI (original Qwen reference rows),
H-02 (self-fitted Llama Instruct), NP-03 (Gemma 3 12B base), and NP-06 (Llama base).
Their complete results remain in `../full_table.json` and `../full_table.md`.

## Outcomes and methods

Both panels show the same eligible latent-intermediate prompt-target swap
trials at strength 1. Left: downstream target-answer success; right: target-name
substring in an incorrect continuation. Correct answers containing the name
count as successes, not leakage. Effects are filtered minus full in percentage
points. Each n is a paired-trial count; one output per lens yields 2n outputs.
The separate concept-use/naming task is not plotted here.

Self-fitted directions are jointly estimated on 32 GSM8K texts. Published full
lenses are held fixed while subtracting a same-position contribution estimated
on 64 WikiText texts; this filtering is approximate. Error bars are paired,
pointwise 95% source-entity bootstrap intervals (2,000 draws), conditional on
one lens/estimate, without multiple-comparison correction. Model eligibility
populations differ; rows are not head-to-head comparisons of models.

## Main-figure caption

Cross-model latent-intermediate swaps at strength 1. Effects are filtered minus
full in percentage points; both panels use the same n paired trials per row.
Self-fitted lenses use jointly estimated full and filtered directions; published
Neuronpedia lenses use approximate filtering with a separately estimated
same-position contribution. Error bars are paired, pointwise 95% source-entity
bootstrap intervals conditional on lens estimates. One row per model/size is
shown using the source/variant selection rule above; all conditions, including
base variants and original Qwen reference panels, are retained in the full table.

Build: `.venv/bin/python scripts/76_core_results_figures.py` (CPU only).

## Exact evaluation counts

| Condition | Model | Paired swap trials (n) | Outputs (2n) | Source-entity clusters |
|---|---|---:|---:|---:|
| NP-01 | Gemma 2 2B IT | 187 | 374 | 58 |
| NP-02 | Gemma 2 9B IT | 222 | 444 | 65 |
| NP-04 | Gemma 3 4B IT | 169 | 338 | 58 |
| NP-05 | Llama 3.1 8B Instruct | 182 | 364 | 57 |
| F-05 | Qwen3-4B | 192 | 384 | 62 |
| F-06 | Qwen3-8B | 189 | 378 | 58 |
| F-04 | Qwen3-14B | 219 | 438 | 64 |
| F-03 | Qwen2.5-7B Instruct | 190 | 380 | 64 |
| F-01 | Mistral-7B Instruct v0.3 | 164 | 328 | 43 |
| F-02 | OLMo-2-7B Instruct | 210 | 420 | 66 |
| H-01 | Gemma 3 12B IT | 222 | 444 | 68 |
