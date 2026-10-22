# Paper-to-code map

The manuscript is deliberately excluded from this repository. This map refers
to its section/topic names, which survive renumbering.

| Paper result | Computation | Recorded input / reproduction |
|---|---|---|
| Jacobian decomposition and pullbacks | `jacobian.component_pullbacks_for_prompt`, `sweeps.run_component_pullbacks` | `configs/paper_components.json`; component `.pt` in reference manifest |
| Fitting-corpus table: five independent fits | `26_tier0_replicates.py`, `domains.py`, `sweeps.run_domain_pullbacks` | `steering_*_tier0_*`, `steering_*_aqua_*`; `reproduce.py tables` |
| Expanded corpus comparisons and new arguments | `26_tier0_replicates.py`, `sweeps.run_causal`; `25_domain_causal.py` regrading | `steering_*_expanded_*`, `steering_*_oscale_abl_*`; `reproduce.py figures` |
| Explicit-argument filtering | `sweeps.run_causal` with component banks | `steering_*_comp_shard{0,1}of2.json`; final figure script |
| Latent-intermediate swaps and frozen-writer/additive controls | `write_controls.py`, `36_write_controls.py` | C22 two complete 55-item JSON runs and frozen expansion plan; `reproduce.py latent` |
| Concept-panel eligibility | `concept_use.build_units`, `run_concept_clean`, `freeze_eligibility` | Frozen panel, clean shards, `eligibility.json` |
| Unit-column/ratio control | `concept_use.swap_deltas`; preset `c36kappa` | `swap_c36kappa_*`; `reproduce.py tables` (complete 661-unit population) |
| Diagonal-weight interpolation, whole-prompt/prefix | `concept_use.run_concept_swap`; presets `c37lambda{prompt,prefix}` | `swap_c37lambda*`; `reproduce.py tables` |
| Natural/applied edit magnitude and spline adjustments | `run_concept_magnitude`; `51_magnitude_report.py` | C38 JSON + `.sites.pt`; `reproduce.py magnitude` |
| Runtime-reflection diagnostic | `run_concept_magmatched` | `magmatched_c38matched*`; magnitude report |
| Relational/leakage gradient prediction | `run_concept_gradgeom`; `55_gradgeom_report.py` | `gradgeom_c39grad*`; `reproduce.py gradients` |
| Float32 central finite differences | `run_concept_gradcheck` | `gradcheck_c39check*`; gradient report |
| Position-pair energy shares | `sweeps.run_energy_map` | `results/energy/energy_*.json` in reference manifest |

Every included artifact is enumerated in `data/reference/manifest.json`, with
both transport and content hashes. See `docs/PAPER_AUDIT.md` before comparing a
historical manuscript cell with an updated report. Original exploratory scripts
remain available for extensions, but their results are not all bundled.

## Subsequent experimental work

| Experiment | Implementation and inputs | Entrypoints |
|---|---|---|
| Expanded latent panel | `latent_panel.py`; `latent-kb.json`, `latent-candidates.json`, `latent-panel.json` | `57_latent_panel.py` (generate, screen, freeze, run, report) |
| Loading and edit-position analyses | `latent_panel.py` loading/profile runners | `58_loading_split.py`, `59_loading_positions.py` |
| Gemma 3 12B and Llama 3.1 8B replication | `cross_model.py`; pinned model configs and `cross_model_filtering.json` | `59_cross_model.py`, `60_cross_model_report.py` |
| Gemma pressure test | `gemma_pressure.py`; frozen `gemma_pressure.json` | `61_gemma_pressure.py`, `fetch_gemma_pressure.py`, `62_gemma_pressure_report.py` |
| Updated manuscript figures | CPU analysis of saved outputs | `63_paper_figures_v3.py` |
| Qwen mechanism figure | Recorded C39 trial margins and restored C37 geometry CSV | `64_mechanism_figure.py` |
| Six additional self-fitted model checkpoints | `cross_model.py`, `filtering_replication.json`, pinned configs | `66_filtering_replication.py`, `67_filtering_replication_report.py` |
| Six published Neuronpedia lenses | `neuronpedia_replication.py`, pinned inventory and WikiText estimation protocol | `72_neuronpedia_replication.py`, `73_neuronpedia_report.py` |
| All-condition results and cross-model figure | Saved paired trial outputs; no new model execution | `75_all_core_results.py`, `76_core_results_figures.py`, `publish_results.py` |

These additions retain their frozen protocols and tests. GPU launchers require
the appropriate model access and an existing Modal deployment; installation
and CPU reanalysis do not launch experiments. The October 21 extension adds
saved trial outputs and JSON metadata to the reference manifest, including all
data inputs for the combined master table. It does not include every auxiliary
tensor or generated file in the ignored `results/` workspace. The curated
public outputs are in [docs/results](results/full_table.md).
Some older figure scripts write to the ignored manuscript figure directory;
`63_paper_figures_v3.py --collect` additionally requires a local manuscript.
The finer position sweep (`65_position_sweep.py`) remains a diagnostic: its
recorded replay check exceeds the frozen threshold, so it is not evidence for
a resolved localization claim in this release.
