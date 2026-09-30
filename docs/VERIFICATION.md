# Release verification

The following checks were completed on Python 3.12 during release preparation.

- Fresh environment installed from `requirements/analysis-tested.txt`, followed
  by an editable install of the extracted anonymous source archive.
- All **49** bundled reference artifacts passed compressed and raw SHA-256
  checks and restored successfully in that extracted copy.
- **164 tests passed; 10 network tests deselected**, with Hugging Face offline
  environment flags enabled. The tests cover Jacobians, steering, grading,
  model-loading guards, revision propagation, archive boundaries and safe
  restoration.
- The extracted copy rebuilt latent counts, corpus replicates, C36 complete-run
  rates and C37 interpolation tables. The headline latent counts matched the
  asserted 15→25 / 13→23 answer hits and 21→5 / 21→9 leakage counts.
- Rebuilding eligibility from both original clean shards exactly matched the
  reference eligibility mapping.
- The original checkout rebuilt the frozen-panel paired report, C38 magnitude
  analysis and C39 gradient report. The reported whole-prompt unadjusted lambda
  effects reproduced: GSM8K R +5.08 / L +4.86, WikiText R +3.12 / L +4.91 nats.
- All seven available scripted figures rebuilt; their generator checked vector
  PDFs, figure width, text bounds, input populations and source counts.
- The release inclusion policy and known-identifier scan passed for source,
  compressed records and Torch metadata. Documentation links and diff whitespace
  checks passed. The ZIP was extracted and checked for corruption/private paths.

These checks establish offline reanalysis and source-package portability on the
tested local platform. They do not establish fresh GPU equivalence, a historical
CUDA environment, Linux CI success, or reproduction of the older 649-unit table
and unsourced manuscript composite. Those limits are listed in PAPER_AUDIT.md.
