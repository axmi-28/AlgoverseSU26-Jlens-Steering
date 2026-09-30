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
