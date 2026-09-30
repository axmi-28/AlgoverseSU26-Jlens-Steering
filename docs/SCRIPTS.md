# Script catalog

Start with `reproduce.py`, `run_paper.py` and `release.py`. Original numeric filenames are stable provenance identifiers. Exploratory scripts need their own generated inputs, which are generally not bundled; they are retained as experiment implementations rather than required paper-reproduction steps.

| Script | Role | Description |
|---|---|---|
| [`00_preflight.py`](../scripts/00_preflight.py) | Paper / validation | Everything checkable before renting a GPU. |
| [`01_check_lens_parity.py`](../scripts/01_check_lens_parity.py) | Paper / validation | Sanity check 1: does our J_bar reproduce the lens Neuronpedia serves? |
| [`02_check_fit_convergence.py`](../scripts/02_check_fit_convergence.py) | Paper / validation | Sanity check 2: does averaging our own J_x converge to the downloaded J_bar? |
| [`03_pullback_demo.py`](../scripts/03_pullback_demo.py) | Exploratory / historical | Capability demo: g_x = J_x^T u_y at scale, and the cost it saves. |
| [`04_jacobian_distribution.py`](../scripts/04_jacobian_distribution.py) | Exploratory / historical | Research question 1: the distribution of prompt-local Jacobians about J_bar. |
| [`05_pullback_distribution.py`](../scripts/05_pullback_distribution.py) | Exploratory / historical | Research question 2: the distribution of local pulled-back directions. |
| [`06_causal_steering.py`](../scripts/06_causal_steering.py) | Exploratory / historical | The causal arm: "how does this vary with causal effects of steering?" |
| [`07_merge_shards.py`](../scripts/07_merge_shards.py) | Paper / validation | Stitch a fanned-out run back together. Runs locally, no GPU. |
| [`08_fetch_results.py`](../scripts/08_fetch_results.py) | Paper / validation | Pull the results volume down, one file at a time, verifying each. |
| [`09_analyze.py`](../scripts/09_analyze.py) | Exploratory / historical | Preliminary distribution tables for RQ1 and RQ2. Local, no GPU. |
| [`10_charts.py`](../scripts/10_charts.py) | Exploratory / historical | Charts for the preliminary results. Static PNGs sized for a Google Doc. |
| [`11_c2_predictors.py`](../scripts/11_c2_predictors.py) | Exploratory / historical | C2 - does the per-prompt geometry predict which prompts steer well? |
| [`12_c2_figures.py`](../scripts/12_c2_figures.py) | Exploratory / historical | Two figures for C2. Static PNGs sized for a document. |
| [`13_scale_comparison.py`](../scripts/13_scale_comparison.py) | Exploratory / historical | Qwen3-8B against Qwen3.6-27B: does model size change the conclusions? |
| [`14_matched_effect.py`](../scripts/14_matched_effect.py) | Exploratory / historical | C3 - do local directions win once the *dose* is matched? |
| [`15_averaging_curve.py`](../scripts/15_averaging_curve.py) | Exploratory / historical | C4 - is the averaged direction's advantage just sample size? |
| [`16_c2_strength_sweep.py`](../scripts/16_c2_strength_sweep.py) | Exploratory / historical | C2 across every steering strength, plus the paper's own predictor as a baseline. |
| [`17_control_figure.py`](../scripts/17_control_figure.py) | Exploratory / historical | One figure: do the prompt-local predictors survive the controls? |
| [`18_linear_response.py`](../scripts/18_linear_response.py) | Exploratory / historical | C5 - is the prompt-local Jacobian the better local model, and out to what dose? |
| [`19_dose_response.py`](../scripts/19_dose_response.py) | Exploratory / historical | C6 - steering local vs averaged directions across dose, at matched collateral. |
| [`20_precondition.py`](../scripts/20_precondition.py) | Exploratory / historical | C7 - does raising the index rescue the prompt-local direction? |
| [`21_predictor_overlap.py`](../scripts/21_predictor_overlap.py) | Exploratory / historical | Do workspace loading and the prompt-Jacobian predictors flag the same prompts? |
| [`22_overlap_figure.py`](../scripts/22_overlap_figure.py) | Exploratory / historical | Figure: are the Jacobian predictors and workspace loading the same signal? |
| [`23_domain_jacobians.py`](../scripts/23_domain_jacobians.py) | Exploratory / historical | C8 -- how much of the averaged Jacobian is the corpus it was fitted on? |
| [`24_domain_figure.py`](../scripts/24_domain_figure.py) | Exploratory / historical | Figure for C8: domain dependence of the averaged Jacobian, against the null. |
| [`25_domain_causal.py`](../scripts/25_domain_causal.py) | Paper / validation | C9 -- does the corpus a J-Lens direction is averaged over change what it steers? |
| [`26_tier0_replicates.py`](../scripts/26_tier0_replicates.py) | Paper / validation | Tier 0 -- an error bar for the domain result. |
| [`27_ordered_scale.py`](../scripts/27_ordered_scale.py) | Exploratory / historical | Held-out test of the ordered-scale hypothesis. |
| [`28_readout.py`](../scripts/28_readout.py) | Exploratory / historical | Does a domain-averaged J_bar READ better, or only write better? |
| [`29_probe_readout.py`](../scripts/29_probe_readout.py) | Exploratory / historical | Can a lens surface a concept the prompt never states? |
| [`30_energy_mechanism.py`](../scripts/30_energy_mechanism.py) | Paper / validation | C16 - does a corpus's Jacobian energy profile explain what its lens does? |
| [`31_remention.py`](../scripts/31_remention.py) | Exploratory / historical | C17 - does the corpus teach the lens that context tokens come back? |
| [`32_two_hop.py`](../scripts/32_two_hop.py) | Exploratory / historical | C19 - writing a latent intermediate, and whether the second hop follows. |
| [`33_loading_components.py`](../scripts/33_loading_components.py) | Exploratory / historical | C20 - is the component result explained by workspace loading? |
| [`34_direction_geometry.py`](../scripts/34_direction_geometry.py) | Exploratory / historical | C21 - how much of each write direction is just the unembedding row u_y? |
| [`35_two_hop_components.py`](../scripts/35_two_hop_components.py) | Paper / validation | C18 x C19 - does dropping the diagonal also improve the LATENT-concept write? |
| [`36_projection_control.py`](../scripts/36_projection_control.py) | Exploratory / historical | C22 - the Gram-Schmidt control: is the mechanism u_y contamination? |
| [`36_write_controls.py`](../scripts/36_write_controls.py) | Paper / validation | C22 preflight, bounded launch, and report. Default action spends no GPU time. |
| [`37_component_readout.py`](../scripts/37_component_readout.py) | Exploratory / historical | C24 - the read side of the component split, measured on our own lens. |
| [`38_fluency.py`](../scripts/38_fluency.py) | Exploratory / historical | C23 - does dropping the diagonal term damage generation? |
| [`39_beta_family.py`](../scripts/39_beta_family.py) | Exploratory / historical | C25 - the unembedding correction as a tunable family, and its closed form. |
| [`40_horizon_gate.py`](../scripts/40_horizon_gate.py) | Exploratory / historical | C26 - is the off-diagonal term one object, or a family ordered by horizon? |
| [`41_horizon_effect.py`](../scripts/41_horizon_effect.py) | Exploratory / historical | C27 - does writing at horizon d move the output d tokens LATER? |
| [`42_write_controls_followup.py`](../scripts/42_write_controls_followup.py) | Paper / validation | Freeze and audit the expanded write-control evaluation, without loading a model. |
| [`43_ablation_effect.py`](../scripts/43_ablation_effect.py) | Exploratory / historical | C28 - the workspace paper's causal measure, applied to our arms. |
| [`44_park_baseline.py`](../scripts/44_park_baseline.py) | Exploratory / historical | C29 - does Park et al.'s causal inner product already do what our correction does? |
| [`45_diffmean_baseline.py`](../scripts/45_diffmean_baseline.py) | Exploratory / historical | C30 - DiffMean/CAA, the supervised baseline, split by template leakage. |
| [`46_index.py`](../scripts/46_index.py) | Exploratory / historical | Generate `results/INDEX.md`: every experiment, its script, and its report. |
| [`47_second_model.py`](../scripts/47_second_model.py) | Exploratory / historical | C31 - does the unembedding correction transfer to a second model? |
| [`48_concept_use.py`](../scripts/48_concept_use.py) | Paper / validation | C32-C35 launcher: presets pin each stage's protocol; nothing here is analysis. |
| [`49_concept_report.py`](../scripts/49_concept_report.py) | Paper / validation | C32-C35 reports. Reads only local copies of the volume files; writes markdown. |
| [`50_lambda_figure.py`](../scripts/50_lambda_figure.py) | Exploratory / historical | C37 figures: the diagonal as a continuum. Reads the runs, writes PNGs + CSV. |
| [`50_paper_figures.py`](../scripts/50_paper_figures.py) | Figure helper (imported) | Audited, publication-size figures from pinned result artifacts. No model/GPU calls. |
| [`51_magnitude_report.py`](../scripts/51_magnitude_report.py) | Paper / validation | C38 report: does realized edit magnitude explain the lambda trend? |
| [`52_submission_figures.py`](../scripts/52_submission_figures.py) | Exploratory / historical | Curated manuscript figures, including C36/C37. CPU-only; raw JSON inputs. |
| [`53_corpus_figures.py`](../scripts/53_corpus_figures.py) | Paper / validation | Two corpus figures, regraded from raw generations. CPU only. |
| [`54_section42_figures.py`](../scripts/54_section42_figures.py) | Exploratory / historical | Three figures corresponding one-to-one to Section 4.2 tables; raw-data rebuild. |
| [`55_gradgeom_report.py`](../scripts/55_gradgeom_report.py) | Paper / validation | C39 report: do relational / leakage gradients explain the lambda trend? |
| [`56_final_paper_figures.py`](../scripts/56_final_paper_figures.py) | Paper / validation | Manuscript-width, claim-led figures. CPU-only rebuild from pinned raw results. |
| [`release.py`](../scripts/release.py) | Release entrypoint | Verify/restore reference data and create a history-free source archive. |
| [`report_write_controls.py`](../scripts/report_write_controls.py) | Paper / validation | Audit and report the bounded latent-write pilot, distinct from C22 projection. |
| [`reproduce.py`](../scripts/reproduce.py) | Release entrypoint | Rebuild paper results on CPU from the verified reference artifacts. |
| [`run_paper.py`](../scripts/run_paper.py) | Release entrypoint | Print reproducible local experiment plans; --execute runs the requested stage. |
