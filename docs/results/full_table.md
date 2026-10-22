# All core J-lens filtering results

Completed position-pair filtering evaluations: 13 model checkpoints, 16 main model/lens evaluations. Rates and paired intervals are derived from saved outputs; no new model calls or GPU work.

IDs: P = original Qwen paper reference; H = initial Gemma/Llama replication; F = broader-model balanced replication; NP = published Neuronpedia-lens replication. Full per-run provenance is listed below.

Full → filtered reports target-answer accuracy in percent. Δ is filtered − full in percentage points. All displayed paired intervals are recomputed with the same source-entity cluster bootstrap (2000 draws, seed 42102), conditional on one fitted lens/diagonal estimate; pointwise and unadjusted for multiple comparisons. The 14 replication rows are checked against their existing summaries. Original Qwen reference intervals can differ slightly from the paper’s 10,000-draw intervals.

Lens provenance: self-fit GSM8K/WikiText rows use jointly fitted full/diagonal/off components (32 fitting texts). Neuronpedia rows hold the published full lens fixed and subtract a separately estimated 64-text WikiText diagonal: this is approximate filtering. Full-fit prompt counts refer to the released lens; D32 is a sensitivity with the same full baseline. Projection proxies remove the unembedding-aligned component and are reported separately.

The main table uses strength 1 and whole-prompt edits. Latent swaps retain native direction-column scales and match filtered edit norms to full. Concept-use/naming comparisons unit-normalize both columns, with per-position/per-layer norm matching. Concept accuracy averages eligible relations within units, then units equally. N is latent items or concept units, not generated tokens or relation queries.

Populations, layer bands, lens-fitting corpus and sampling differ across campaigns; compare full vs filtered within each row. The original Qwen panel uses 194 latent items and 661 country-heavy concept units; later replications select up to 40 concept units per type. Original Qwen paper-reference and later Qwen control rows are distinct evaluations and are not pooled. Naming accuracy and incorrect-name emission are separate endpoints; emission reduction alone does not establish improved use.

Gemma pressure-test validation is exploratory: source groups are separated from development selection, but earlier aggregate/some latent outcomes were already observed. All six evaluated validation settings are retained, including unsuccessful settings. Its projection row is a proxy, and its native-column concept controls differ from the main unit-column comparison.

Coverage: Qwen3-1.7B and Qwen3.6-27B have other domain/geometry or local-vs-averaged experiments, but no completed paired latent-plus-concept position-filtering replication was located for this table. They are untested for these endpoints, not negative results. The configured Gemma 2 2B base checkpoint likewise has no completed core pair here.

## Main comparison · strength 1

| ID | Model | J-lens / filtering | Fit texts: full / D | Latent N | Latent full → filtered (%); Δ pp [95% CI] | Concept N | Use full → filtered (%); Δ pp [95% CI] |
|---|---|---|---|---|---|---|---|
| P-GSM | qwen3-8b | Self-fit GSM8K / exact components | 32 / 32 | 194 | 32.0 → 47.4; +15.5 [+8.8, +21.7] | 661 | 66.8 → 74.9; +8.2 [+3.1, +13.7] |
| P-WIKI | qwen3-8b | Self-fit WikiText / exact components | 32 / 32 | 194 | 35.1 → 49.5; +14.4 [+7.7, +20.7] | 661 | 60.6 → 77.1; +16.4 [+12.6, +20.5] |
| H-01 | gemma-3-12b-it | Self-fit GSM8K / exact components | 32 / 32 | 222 | 30.6 → 30.2; -0.5 [-6.0, +5.3] | 120 | 24.2 → 18.8; -5.4 [-11.2, +0.5] |
| H-02 | llama-3.1-8b-instruct | Self-fit GSM8K / exact components | 32 / 32 | 183 | 21.9 → 36.6; +14.8 [+9.6, +19.5] | 89 | 21.0 → 33.3; +12.4 [+6.3, +19.9] |
| F-01 | mistral-7b-instruct-v0.3 | Self-fit GSM8K / exact components | 32 / 32 | 164 | 25.0 → 44.5; +19.5 [+12.5, +26.5] | 106 | 29.7 → 38.7; +9.0 [+3.7, +15.6] |
| F-02 | olmo-2-1124-7b-instruct | Self-fit GSM8K / exact components | 32 / 32 | 210 | 30.5 → 40.5; +10.0 [+1.5, +18.9] | 120 | 36.7 → 32.9; -3.7 [-14.7, +7.5] |
| F-03 | qwen2.5-7b-instruct | Self-fit GSM8K / exact components | 32 / 32 | 190 | 44.2 → 40.5; -3.7 [-10.0, +2.3] | 120 | 35.1 → 26.0; -9.2 [-15.6, -2.3] |
| F-04 | qwen3-14b | Self-fit GSM8K / exact components | 32 / 32 | 219 | 49.3 → 54.8; +5.5 [+0.5, +10.0] | 94 | 40.6 → 43.6; +3.0 [-3.2, +10.5] |
| F-05 | qwen3-4b | Self-fit GSM8K / exact components | 32 / 32 | 192 | 27.6 → 35.9; +8.3 [+2.3, +14.7] | 120 | 37.9 → 42.8; +4.9 [-1.2, +11.1] |
| F-06 | qwen3-8b | Self-fit GSM8K / exact components | 32 / 32 | 189 | 31.7 → 47.6; +15.9 [+9.6, +21.9] | 120 | 51.4 → 57.2; +5.8 [-0.2, +12.9] |
| NP-01 | gemma-2-2b-it | Neuronpedia WikiText + estimated D64 | 337 / 64 | 187 | 16.6 → 33.2; +16.6 [+10.1, +23.9] | 120 | 21.4 → 34.7; +13.3 [+7.4, +19.6] |
| NP-02 | gemma-2-9b-it | Neuronpedia WikiText + estimated D64 | 453 / 64 | 222 | 31.1 → 39.2; +8.1 [+4.1, +12.3] | 120 | 41.4 → 50.6; +9.2 [+4.4, +13.8] |
| NP-03 | gemma-3-12b-pt | Neuronpedia WikiText + estimated D64 | 775 / 64 | 207 | 26.6 → 29.0; +2.4 [-3.5, +8.7] | 120 | 24.2 → 20.1; -4.0 [-9.3, +0.7] |
| NP-04 | gemma-3-4b-it | Neuronpedia WikiText + estimated D64 | 546 / 64 | 169 | 25.4 → 34.3; +8.9 [+4.1, +14.0] | 101 | 19.8 → 21.5; +1.7 [-4.4, +7.5] |
| NP-05 | llama-3.1-8b-instruct | Neuronpedia WikiText + estimated D64 | 461 / 64 | 182 | 15.4 → 27.5; +12.1 [+6.5, +17.6] | 89 | 19.9 → 29.4; +9.6 [+2.6, +19.8] |
| NP-06 | llama-3.1-8b | Neuronpedia WikiText + estimated D64 | 539 / 64 | 175 | 17.1 → 32.0; +14.9 [+8.1, +21.9] | 115 | 23.5 → 29.6; +6.1 [-0.1, +13.8] |

## Naming versus downstream use · same concept units, strength 1

| ID | Model | Concept N | Naming full → filtered (%) | Naming Δ pp [95% CI] | Use full → filtered (%) | Use Δ pp [95% CI] | Incorrect name on use full → filtered (%) |
|---|---|---|---|---|---|---|---|
| P-GSM | qwen3-8b | 661 | 95.9 → 86.5 | -9.4 [-14.1, -5.4] | 66.8 → 74.9 | +8.2 [+3.1, +13.7] | 21.5 → 5.7 |
| P-WIKI | qwen3-8b | 661 | 95.5 → 92.3 | -3.2 [-6.0, -0.5] | 60.6 → 77.1 | +16.4 [+12.6, +20.5] | 30.1 → 10.1 |
| H-01 | gemma-3-12b-it | 120 | 60.8 → 19.2 | -41.7 [-53.3, -31.4] | 24.2 → 18.8 | -5.4 [-11.2, +0.5] | 21.2 → 4.4 |
| H-02 | llama-3.1-8b-instruct | 89 | 82.0 → 67.4 | -14.6 [-23.2, -6.3] | 21.0 → 33.3 | +12.4 [+6.3, +19.9] | 48.7 → 28.3 |
| F-01 | mistral-7b-instruct-v0.3 | 106 | 90.6 → 84.0 | -6.6 [-15.4, +1.1] | 29.7 → 38.7 | +9.0 [+3.7, +15.6] | 58.2 → 35.8 |
| F-02 | olmo-2-1124-7b-instruct | 120 | 98.3 → 40.8 | -57.5 [-69.5, -43.8] | 36.7 → 32.9 | -3.7 [-14.7, +7.5] | 43.9 → 3.3 |
| F-03 | qwen2.5-7b-instruct | 120 | 85.8 → 40.0 | -45.8 [-60.6, -29.2] | 35.1 → 26.0 | -9.2 [-15.6, -2.3] | 21.8 → 0.0 |
| F-04 | qwen3-14b | 94 | 93.6 → 63.8 | -29.8 [-44.4, -14.8] | 40.6 → 43.6 | +3.0 [-3.2, +10.5] | 40.1 → 11.5 |
| F-05 | qwen3-4b | 120 | 93.3 → 73.3 | -20.0 [-29.0, -10.9] | 37.9 → 42.8 | +4.9 [-1.2, +11.1] | 38.5 → 27.4 |
| F-06 | qwen3-8b | 120 | 91.7 → 78.3 | -13.3 [-20.9, -6.4] | 51.4 → 57.2 | +5.8 [-0.2, +12.9] | 29.7 → 12.1 |
| NP-01 | gemma-2-2b-it | 120 | 93.3 → 91.7 | -1.7 [-6.2, +1.8] | 21.4 → 34.7 | +13.3 [+7.4, +19.6] | 68.5 → 50.4 |
| NP-02 | gemma-2-9b-it | 120 | 83.3 → 80.0 | -3.3 [-9.6, +2.5] | 41.4 → 50.6 | +9.2 [+4.4, +13.8] | 33.6 → 21.0 |
| NP-03 | gemma-3-12b-pt | 120 | 93.3 → 53.3 | -40.0 [-49.3, -28.6] | 24.2 → 20.1 | -4.0 [-9.3, +0.7] | 58.3 → 22.5 |
| NP-04 | gemma-3-4b-it | 101 | 77.2 → 40.6 | -36.6 [-45.3, -26.7] | 19.8 → 21.5 | +1.7 [-4.4, +7.5] | 30.7 → 11.2 |
| NP-05 | llama-3.1-8b-instruct | 89 | 79.8 → 62.9 | -16.9 [-26.6, -7.7] | 19.9 → 29.4 | +9.6 [+2.6, +19.8] | 47.9 → 28.8 |
| NP-06 | llama-3.1-8b | 115 | 80.0 → 66.1 | -13.9 [-22.0, -6.0] | 23.5 → 29.6 | +6.1 [-0.1, +13.8] | 45.9 → 35.9 |

## Both latent strengths and complete paired metrics

| ID | Model | Task / strength | N | Source clusters | Accuracy full → filtered (%) | Accuracy Δ pp [95% CI] | Incorrect name full → filtered (%) | Incorrect-name Δ pp [95% CI] |
|---|---|---|---|---|---|---|---|---|
| P-GSM | qwen3-8b | latent / 0.5 | 194 | 58 | 37.6 → 26.8 | -10.8 [-15.6, -6.1] | 10.8 → 4.6 | -6.2 [-9.7, -3.1] |
| P-GSM | qwen3-8b | latent / 1 | 194 | 58 | 32.0 → 47.4 | +15.5 [+8.8, +21.7] | 34.0 → 8.8 | -25.3 [-31.7, -18.8] |
| P-GSM | qwen3-8b | concept use / 1 | 661 | 30 | 66.8 → 74.9 | +8.2 [+3.1, +13.7] | 21.5 → 5.7 | -15.8 [-20.2, -11.9] |
| P-WIKI | qwen3-8b | latent / 0.5 | 194 | 58 | 43.3 → 46.9 | +3.6 [-1.0, +8.6] | 14.9 → 7.2 | -7.7 [-12.8, -3.7] |
| P-WIKI | qwen3-8b | latent / 1 | 194 | 58 | 35.1 → 49.5 | +14.4 [+7.7, +20.7] | 35.1 → 13.4 | -21.6 [-29.3, -14.5] |
| P-WIKI | qwen3-8b | concept use / 1 | 661 | 30 | 60.6 → 77.1 | +16.4 [+12.6, +20.5] | 30.1 → 10.1 | -20.0 [-23.5, -16.4] |
| H-01 | gemma-3-12b-it | latent / 0.5 | 222 | 68 | 23.0 → 18.0 | -5.0 [-8.3, -1.4] | 6.3 → 1.4 | -5.0 [-7.6, -2.4] |
| H-01 | gemma-3-12b-it | latent / 1 | 222 | 68 | 30.6 → 30.2 | -0.5 [-6.0, +5.3] | 15.3 → 5.0 | -10.4 [-15.4, -5.4] |
| H-01 | gemma-3-12b-it | concept use / 1 | 120 | 33 | 24.2 → 18.8 | -5.4 [-11.2, +0.5] | 21.2 → 4.4 | -16.8 [-24.6, -9.4] |
| H-02 | llama-3.1-8b-instruct | latent / 0.5 | 183 | 57 | 33.9 → 42.6 | +8.7 [+4.3, +13.0] | 29.5 → 16.4 | -13.1 [-18.7, -7.3] |
| H-02 | llama-3.1-8b-instruct | latent / 1 | 183 | 57 | 21.9 → 36.6 | +14.8 [+9.6, +19.5] | 47.0 → 27.3 | -19.7 [-27.0, -13.1] |
| H-02 | llama-3.1-8b-instruct | concept use / 1 | 89 | 22 | 21.0 → 33.3 | +12.4 [+6.3, +19.9] | 48.7 → 28.3 | -20.4 [-29.4, -12.0] |
| F-01 | mistral-7b-instruct-v0.3 | latent / 0.5 | 164 | 43 | 46.3 → 51.2 | +4.9 [+0.0, +9.5] | 30.5 → 15.9 | -14.6 [-20.6, -9.2] |
| F-01 | mistral-7b-instruct-v0.3 | latent / 1 | 164 | 43 | 25.0 → 44.5 | +19.5 [+12.5, +26.5] | 62.2 → 37.8 | -24.4 [-32.1, -16.9] |
| F-01 | mistral-7b-instruct-v0.3 | concept use / 1 | 106 | 27 | 29.7 → 38.7 | +9.0 [+3.7, +15.6] | 58.2 → 35.8 | -22.3 [-33.2, -12.5] |
| F-02 | olmo-2-1124-7b-instruct | latent / 0.5 | 210 | 66 | 40.0 → 6.7 | -33.3 [-42.9, -23.3] | 22.4 → 1.9 | -20.5 [-27.2, -14.6] |
| F-02 | olmo-2-1124-7b-instruct | latent / 1 | 210 | 66 | 30.5 → 40.5 | +10.0 [+1.5, +18.9] | 48.6 → 7.1 | -41.4 [-49.5, -34.0] |
| F-02 | olmo-2-1124-7b-instruct | concept use / 1 | 120 | 33 | 36.7 → 32.9 | -3.7 [-14.7, +7.5] | 43.9 → 3.3 | -40.6 [-55.0, -25.7] |
| F-03 | qwen2.5-7b-instruct | latent / 0.5 | 190 | 64 | 23.7 → 12.6 | -11.1 [-16.7, -5.9] | 3.2 → 0.5 | -2.6 [-4.8, -0.6] |
| F-03 | qwen2.5-7b-instruct | latent / 1 | 190 | 64 | 44.2 → 40.5 | -3.7 [-10.0, +2.3] | 14.2 → 4.2 | -10.0 [-14.9, -5.5] |
| F-03 | qwen2.5-7b-instruct | concept use / 1 | 120 | 31 | 35.1 → 26.0 | -9.2 [-15.6, -2.3] | 21.8 → 0.0 | -21.8 [-32.9, -11.0] |
| F-04 | qwen3-14b | latent / 0.5 | 219 | 64 | 38.4 → 32.0 | -6.4 [-10.6, -2.0] | 5.9 → 3.2 | -2.7 [-5.0, -0.9] |
| F-04 | qwen3-14b | latent / 1 | 219 | 64 | 49.3 → 54.8 | +5.5 [+0.5, +10.0] | 24.2 → 10.5 | -13.7 [-18.4, -9.0] |
| F-04 | qwen3-14b | concept use / 1 | 94 | 14 | 40.6 → 43.6 | +3.0 [-3.2, +10.5] | 40.1 → 11.5 | -28.5 [-42.1, -15.8] |
| F-05 | qwen3-4b | latent / 0.5 | 192 | 62 | 35.4 → 30.7 | -4.7 [-10.5, +1.4] | 18.8 → 8.9 | -9.9 [-15.5, -5.0] |
| F-05 | qwen3-4b | latent / 1 | 192 | 62 | 27.6 → 35.9 | +8.3 [+2.3, +14.7] | 40.6 → 20.8 | -19.8 [-26.3, -12.5] |
| F-05 | qwen3-4b | concept use / 1 | 120 | 29 | 37.9 → 42.8 | +4.9 [-1.2, +11.1] | 38.5 → 27.4 | -11.1 [-20.7, -2.1] |
| F-06 | qwen3-8b | latent / 0.5 | 189 | 58 | 38.6 → 26.5 | -12.2 [-17.7, -7.1] | 9.5 → 4.2 | -5.3 [-8.4, -2.6] |
| F-06 | qwen3-8b | latent / 1 | 189 | 58 | 31.7 → 47.6 | +15.9 [+9.6, +21.9] | 34.4 → 8.5 | -25.9 [-32.1, -19.7] |
| F-06 | qwen3-8b | concept use / 1 | 120 | 29 | 51.4 → 57.2 | +5.8 [-0.2, +12.9] | 29.7 → 12.1 | -17.6 [-27.6, -8.7] |
| NP-01 | gemma-2-2b-it | latent / 0.5 | 187 | 58 | 35.8 → 45.5 | +9.6 [+5.7, +13.7] | 39.6 → 28.3 | -11.2 [-15.3, -7.3] |
| NP-01 | gemma-2-2b-it | latent / 1 | 187 | 58 | 16.6 → 33.2 | +16.6 [+10.1, +23.9] | 61.0 → 40.6 | -20.3 [-25.3, -15.0] |
| NP-01 | gemma-2-2b-it | concept use / 1 | 120 | 29 | 21.4 → 34.7 | +13.3 [+7.4, +19.6] | 68.5 → 50.4 | -18.1 [-24.6, -11.8] |
| NP-02 | gemma-2-9b-it | latent / 0.5 | 222 | 65 | 35.6 → 41.4 | +5.9 [+1.8, +10.4] | 19.4 → 9.0 | -10.4 [-15.3, -6.2] |
| NP-02 | gemma-2-9b-it | latent / 1 | 222 | 65 | 31.1 → 39.2 | +8.1 [+4.1, +12.3] | 42.8 → 25.7 | -17.1 [-23.8, -11.3] |
| NP-02 | gemma-2-9b-it | concept use / 1 | 120 | 31 | 41.4 → 50.6 | +9.2 [+4.4, +13.8] | 33.6 → 21.0 | -12.6 [-17.9, -6.8] |
| NP-03 | gemma-3-12b-pt | latent / 0.5 | 207 | 60 | 45.4 → 44.9 | -0.5 [-5.4, +4.8] | 13.5 → 4.3 | -9.2 [-13.4, -5.2] |
| NP-03 | gemma-3-12b-pt | latent / 1 | 207 | 60 | 26.6 → 29.0 | +2.4 [-3.5, +8.7] | 31.9 → 10.6 | -21.3 [-27.9, -14.8] |
| NP-03 | gemma-3-12b-pt | concept use / 1 | 120 | 32 | 24.2 → 20.1 | -4.0 [-9.3, +0.7] | 58.3 → 22.5 | -35.8 [-44.8, -25.3] |
| NP-04 | gemma-3-4b-it | latent / 0.5 | 169 | 58 | 46.2 → 46.2 | +0.0 [-5.5, +5.6] | 17.8 → 9.5 | -8.3 [-12.9, -4.1] |
| NP-04 | gemma-3-4b-it | latent / 1 | 169 | 58 | 25.4 → 34.3 | +8.9 [+4.1, +14.0] | 34.9 → 12.4 | -22.5 [-28.9, -16.4] |
| NP-04 | gemma-3-4b-it | concept use / 1 | 101 | 28 | 19.8 → 21.5 | +1.7 [-4.4, +7.5] | 30.7 → 11.2 | -19.5 [-27.0, -11.9] |
| NP-05 | llama-3.1-8b-instruct | latent / 0.5 | 182 | 57 | 34.6 → 44.5 | +9.9 [+3.7, +16.9] | 31.3 → 17.0 | -14.3 [-20.3, -8.6] |
| NP-05 | llama-3.1-8b-instruct | latent / 1 | 182 | 57 | 15.4 → 27.5 | +12.1 [+6.5, +17.6] | 46.2 → 34.1 | -12.1 [-20.5, -3.5] |
| NP-05 | llama-3.1-8b-instruct | concept use / 1 | 89 | 22 | 19.9 → 29.4 | +9.6 [+2.6, +19.8] | 47.9 → 28.8 | -19.1 [-29.6, -10.9] |
| NP-06 | llama-3.1-8b | latent / 0.5 | 175 | 54 | 33.7 → 46.3 | +12.6 [+6.8, +18.5] | 37.7 → 23.4 | -14.3 [-20.8, -7.9] |
| NP-06 | llama-3.1-8b | latent / 1 | 175 | 54 | 17.1 → 32.0 | +14.9 [+8.1, +21.9] | 49.1 → 32.0 | -17.1 [-24.0, -10.5] |
| NP-06 | llama-3.1-8b | concept use / 1 | 115 | 26 | 23.5 → 29.6 | +6.1 [-0.1, +13.8] | 45.9 → 35.9 | -10.0 [-18.3, -3.0] |

## Published-lens sensitivity · diagonal estimated on 32 vs 64 texts

| ID | Model | Task / strength | N | Published full (%) | Filtered D32 (%) | Filtered D64 (%) | D32 Δ pp [95% CI] | D64 Δ pp [95% CI] |
|---|---|---|---|---|---|---|---|---|
| NP-01 | gemma-2-2b-it | latent / 0.5 | 187 | 35.8 | 44.9 | 45.5 | +9.1 [+5.4, +13.0] | +9.6 [+5.7, +13.7] |
| NP-01 | gemma-2-2b-it | latent / 1 | 187 | 16.6 | 34.2 | 33.2 | +17.6 [+11.6, +24.6] | +16.6 [+10.1, +23.9] |
| NP-01 | gemma-2-2b-it | concept use / 1 | 120 | 21.4 | 34.4 | 34.7 | +13.1 [+7.0, +19.2] | +13.3 [+7.4, +19.6] |
| NP-02 | gemma-2-9b-it | latent / 0.5 | 222 | 35.6 | 41.4 | 41.4 | +5.9 [+2.0, +10.3] | +5.9 [+1.8, +10.4] |
| NP-02 | gemma-2-9b-it | latent / 1 | 222 | 31.1 | 37.8 | 39.2 | +6.8 [+2.7, +11.0] | +8.1 [+4.1, +12.3] |
| NP-02 | gemma-2-9b-it | concept use / 1 | 120 | 41.4 | 49.0 | 50.6 | +7.6 [+3.5, +12.7] | +9.2 [+4.4, +13.8] |
| NP-03 | gemma-3-12b-pt | latent / 0.5 | 207 | 45.4 | 44.9 | 44.9 | -0.5 [-5.7, +5.1] | -0.5 [-5.4, +4.8] |
| NP-03 | gemma-3-12b-pt | latent / 1 | 207 | 26.6 | 30.4 | 29.0 | +3.9 [-1.6, +9.9] | +2.4 [-3.5, +8.7] |
| NP-03 | gemma-3-12b-pt | concept use / 1 | 120 | 24.2 | 23.1 | 20.1 | -1.1 [-6.3, +3.8] | -4.0 [-9.3, +0.7] |
| NP-04 | gemma-3-4b-it | latent / 0.5 | 169 | 46.2 | 46.2 | 46.2 | +0.0 [-5.7, +5.6] | +0.0 [-5.5, +5.6] |
| NP-04 | gemma-3-4b-it | latent / 1 | 169 | 25.4 | 36.7 | 34.3 | +11.2 [+5.5, +17.6] | +8.9 [+4.1, +14.0] |
| NP-04 | gemma-3-4b-it | concept use / 1 | 101 | 19.8 | 21.8 | 21.5 | +2.0 [-4.4, +9.0] | +1.7 [-4.4, +7.5] |
| NP-05 | llama-3.1-8b-instruct | latent / 0.5 | 182 | 34.6 | 41.8 | 44.5 | +7.1 [+1.1, +13.0] | +9.9 [+3.7, +16.9] |
| NP-05 | llama-3.1-8b-instruct | latent / 1 | 182 | 15.4 | 26.4 | 27.5 | +11.0 [+4.7, +17.1] | +12.1 [+6.5, +17.6] |
| NP-05 | llama-3.1-8b-instruct | concept use / 1 | 89 | 19.9 | 35.2 | 29.4 | +15.4 [+7.9, +24.9] | +9.6 [+2.6, +19.8] |
| NP-06 | llama-3.1-8b | latent / 0.5 | 175 | 33.7 | 48.0 | 46.3 | +14.3 [+8.2, +20.9] | +12.6 [+6.8, +18.5] |
| NP-06 | llama-3.1-8b | latent / 1 | 175 | 17.1 | 36.0 | 32.0 | +18.9 [+12.6, +25.1] | +14.9 [+8.1, +21.9] |
| NP-06 | llama-3.1-8b | concept use / 1 | 115 | 23.5 | 33.3 | 29.6 | +9.9 [+4.0, +16.1] | +6.1 [-0.1, +13.8] |

## Gemma 3 12B IT follow-up · validation only

| Model | Setting: corpus / layers / strength / columns | Selection | Method | Latent N | Latent full → filtered (%); Δ pp [95% CI] | Concept N | Concept full → filtered (%); Δ pp [95% CI] |
|---|---|---|---|---|---|---|---|
| gemma-3-12b-it | gsm8k/early/2/native | Dev-selected | Self-fit GSM8K (32 texts) / exact components | 104 | 47.1 → 46.2; -1.0 [-6.8, +3.4] | 60 | 59.2 → 59.2; +0.0 [-6.1, +4.9] |
| gemma-3-12b-it | gsm8k/original/1/native | Original fixed control | Self-fit GSM8K (32 texts) / exact components | 104 | 30.8 → 27.9; -2.9 [-9.9, +3.8] | 60 | 18.9 → 13.9; -5.0 [-14.1, +1.6] |
| gemma-3-12b-it | gsm8k/original/1/unit | Original fixed control | Self-fit GSM8K (32 texts) / exact components | 104 | 28.8 → 25.0; -3.8 [-9.1, +1.5] | 60 | 23.9 → 10.3; -13.6 [-25.2, -5.9] |
| gemma-3-12b-it | released/original/1/native | Original fixed control | Neuronpedia WikiText (844 texts) / unembedding-projection proxy | 104 | 27.9 → 18.3; -9.6 [-16.5, -2.3] | 60 | 24.7 → 17.8; -6.9 [-18.6, +1.0] |
| gemma-3-12b-it | wikitext_a/early/1.5/unit | Dev-selected | Self-fit WikiText (32 texts) / exact components | 104 | 44.2 → 44.2; +0.0 [-6.7, +6.0] | 60 | 55.0 → 60.8; +5.8 [+0.8, +12.1] |
| gemma-3-12b-it | wikitext_a/original/1/native | Original fixed control | Self-fit WikiText (32 texts) / exact components | 104 | 26.9 → 29.8; +2.9 [-4.1, +9.0] | 60 | 28.1 → 15.6; -12.5 [-21.0, -6.0] |

## Published-lens projection controls · explicit non-numerical swaps

| Model | Method | N | Full correct | Projected correct | Full → projected (%) | Δ pp |
|---|---|---|---|---|---|---|
| qwen3-4b | published WikiText / unembedding-projection proxy (beta=1) | 99 | 18 | 40 | 18.2 → 40.4 | +22.2 |
| qwen3-8b | published WikiText / unembedding-projection proxy (beta=1) | 105 | 20 | 47 | 19.0 → 44.8 | +25.7 |

## Original Qwen 3 8B prefix-only placement control · strength 1

| Model | Self-fit lens | Concept N | Use full → filtered (%) | Use Δ pp [95% CI] | Incorrect name full → filtered (%) |
|---|---|---|---|---|---|
| qwen3-8b | gsm8k | 661 | 55.4 → 47.9 | -7.5 [-10.9, -4.4] | 0.9 → 0.8 |
| qwen3-8b | wikitext_a | 661 | 76.9 → 78.1 | +1.2 [-0.5, +3.2] | 1.4 → 2.0 |

## Run provenance

| ID | Model revision | Edited layers | Source artifacts |
|---|---|---|---|
| P-GSM | See source manifests | 13, 16, 19, 22, 25, 28, 31 | results/c22/write_controls_dac7603b1af57dbd.json; results/latent/latent_controls_gsm8k_80a6934224f0ac55.json; results/concept/swap_c37lambdaprompt_qwen3-8b_034da26ce9_shard[0–3]of4.json |
| P-WIKI | See source manifests | 13, 16, 19, 22, 25, 28, 31 | results/c22/write_controls_6fff00659415ddf5.json; results/latent/latent_controls_wikitext_a_393f229cc9c869be.json; results/concept/swap_c37lambdaprompt_qwen3-8b_034da26ce9_shard[0–3]of4.json |
| H-01 | 96b6f1eccf38110c56df3a15bffe176da04bfd80 | 17, 21, 26, 30, 34, 38, 42 | results/cross_model/gemma-3-12b-it/22b06e8c0089 |
| H-02 | 0e9e39f249a16976918f6564b8830bc894c89659 | 12, 14, 17, 19, 22, 25, 27 | results/cross_model/llama-3.1-8b-instruct/19b48f435ca8 |
| F-01 | c170c708c41dac9275d15a8fff4eca08d52bab71 | 12, 14, 17, 19, 22, 25, 27 | results/filtering_replication/mistral-7b-instruct-v0.3/5d0378eb34d3 |
| F-02 | 470b1fba1ae01581f270116362ee4aa1b97f4c84 | 12, 14, 17, 19, 22, 25, 27 | results/filtering_replication/olmo-2-1124-7b-instruct/5b97caf5275b |
| F-03 | a09a35458c702b33eeacc393d103063234e8bc28 | 10, 12, 15, 17, 19, 22, 24 | results/filtering_replication/qwen2.5-7b-instruct/f3b780e2c198 |
| F-04 | 40c069824f4251a91eefaf281ebe4c544efd3e18 | 14, 18, 21, 25, 28, 31, 35 | results/filtering_replication/qwen3-14b/f35c5fb3d674 |
| F-05 | 1cfa9a7208912126459214e8b04321603b3df60c | 13, 16, 19, 22, 25, 28, 31 | results/filtering_replication/qwen3-4b/144f15df1fd0 |
| F-06 | b968826d9c46dd6066d109eabc6255188de91218 | 13, 16, 19, 22, 25, 28, 31 | results/filtering_replication/qwen3-8b/592cdb16c2cc |
| NP-01 | 299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8 | 9, 11, 14, 16, 18, 20, 22 | results/neuronpedia_replication/gemma-2-2b-it-np/c2382ae25c35 |
| NP-02 | 11c9b309abf73637e4b6f9a3fa1e92e615547819 | 15, 19, 22, 26, 29, 33, 36 | results/neuronpedia_replication/gemma-2-9b-it-np/d632d955f641 |
| NP-03 | 295efb63d01a7017928f273a94ebb86105c9526f | 17, 21, 26, 30, 34, 38, 42 | results/neuronpedia_replication/gemma-3-12b-pt-np/788be0987a8e |
| NP-04 | 093f9f388b31de276ce2de164bdc2081324b9767 | 12, 15, 18, 21, 24, 26, 29 | results/neuronpedia_replication/gemma-3-4b-it-np/0f7ee6bdd66d |
| NP-05 | 0e9e39f249a16976918f6564b8830bc894c89659 | 12, 14, 17, 19, 22, 25, 27 | results/neuronpedia_replication/llama-3.1-8b-instruct-np/08e31a2fa1c2 |
| NP-06 | d04e592bb4f6aa9cfee91e2e20afa771667e1d4b | 12, 14, 17, 19, 22, 25, 27 | results/neuronpedia_replication/llama-3.1-8b-np/ac6f9f78ca05 |

## Rebuild

`.venv/bin/python scripts/75_all_core_results.py`

Sources include the original Qwen C22/C40/C37 artifacts, completed cross_model, filtering_replication and neuronpedia_replication runs, Gemma pressure validation, and the C31 published-lens projection report. The JSON provenance file hashes every artifact read.
