# Camera-ready source and results release

The October 21, 2026 release accompanies the paper *Position-Pair Filtering
Separates Concept Use from Name Leakage in J-lens Swaps*, accepted at ACCV 2026.
It is de-anonymized and retains the existing public Git history.

## Included material

- Research code, pinned model configurations, frozen evaluation definitions,
  both replication protocols and offline tests.
- 217 checksummed reference artifacts: the original 49-file bundle plus saved
  cross-model trial outputs, campaign JSON metadata, the mechanism CSV and other
  data inputs needed for the combined results table.
- The curated [public results snapshot](results/full_table.md), vector/raster
  figures, plotting statistics, source provenance and snapshot checksums.
- Both optional remote entrypoints, `modal_app.py` and `modal_replication.py`.

The manifest preserves historical input/output bytes; gzip is transport only.
`restore` refuses to overwrite differing local results. Model weights, downloaded
published lenses and the complete auxiliary tensor banks are not included.
Reanalysis of saved generations does not repeat a GPU experiment or establish
variation across independent lens fits. See [reproduction instructions](REPRODUCING.md).

## Verify and package

```bash
python scripts/release.py verify
python scripts/release.py restore
python scripts/release.py audit
python scripts/publish_results.py verify
python -m pytest -q
python scripts/release.py archive
```

The default archive is `dist/jlens-steering-camera-ready.zip`, with fixed
archive timestamps, a neutral root directory, SHA256SUMS and no Git metadata.
It includes only the explicit allowlist in `scripts/release.py`. Local
manuscripts, review correspondence, notes, logs, credentials and uncurated
results remain excluded. The public snapshot under `docs/results/` is
intentionally included; the general `output/` tree remains ignored.

The scanner checks common absolute-path and credential patterns, including
decompressed reference records and Torch metadata. It does not guarantee that
a public repository or this identified author/citation information is anonymous.
The archive's earlier anonymous-review name/workflow is historical.

The complete source/reference archive is larger than the 100 MB conference
supplementary-file limit. It is intended for distribution through the code host,
not direct upload as an ACCV supplementary file. A conference supplement is
optional and must be curated and checked separately; nothing in this workflow
uploads to OpenReview.

## Updating the published snapshot

```bash
python scripts/75_all_core_results.py
python scripts/76_core_results_figures.py
python scripts/publish_results.py build
python scripts/publish_results.py verify --rebuilt
```

Do not edit derived point estimates by hand. The full table retains all 16
model/lens conditions. The main figure displays 11 distinct model/size entries,
preferring instruction-tuned variants, published lenses for identical checkpoints,
and the newer Qwen3-8B replication. Selection uses no outcome values and does
not average or discard the other conditions.

Original project code is MIT-licensed. Third-party data, models and dependencies
retain their upstream terms; see [attributions](../THIRD_PARTY_NOTICES.md).
