# Data and provenance

`heldout/` contains original frozen evaluation definitions: ordered-scale,
depth-ladder and concept-use. The paper's primary concept panel is
`concept-use.json`; eligibility additionally depends on the clean model run.

`protocols/write_controls_expansion_plan.json` preserves the latent panel's
original frozen protocol, including pilot items, expected manifests and paired
contrasts. Its historical status string describes when the plan was frozen;
completion is checked against the actual saved run payloads.

`reference/manifest.json` lists 49 losslessly compressed files. Each row records
the original relative destination, purpose, byte count, SHA-256 of the original
bytes, and SHA-256 of the gzip transport. No original numerical data or manifest
was rewritten for cleanup. `python scripts/release.py restore` recreates the
original files and rejects conflicts. Only known, checksummed tensor artifacts
should be loaded by the legacy report code.

The bundle contains the final scripted-figure inputs, complete C36/C37/C38/C39
outputs required by paper analyses, per-site magnitude tensors, clean eligibility,
the component bank, component-fit corpus snapshot, position-pair energy outputs,
latent pilot/reference outputs, and the two pinned upstream evaluation files.
It excludes base-model weights, most exploratory outputs and redundant generated
plots. Gradient reports use the saved gradient summaries; the large optional
per-site gradient tensors are not required or included.

The component-fit snapshot preserves text from the original materialized corpus
cache, including auxiliary corpus groups. Each group has 32 documents. Corpus
assembly and upstream dataset identifiers are defined in `src/jsteer/domains.py`.
The snapshot was not retokenized or normalized. This is source data from upstream
datasets, not newly authored evaluation text. See THIRD_PARTY_NOTICES.md.

The upstream flexible-generalization and probe-swap JSON files come from
`anthropics/jacobian-lens` commit `581d398613e5602a5af361e1c34d3a92ea82ba8e`.
`src/jsteer/data.py` can refetch them, but restoration supports offline use.
Model weights and general corpus downloads belong in external caches. Original
corpus-replicate dataset revisions were not recorded; fresh streams are not
claimed to be the exact historical text samples.
