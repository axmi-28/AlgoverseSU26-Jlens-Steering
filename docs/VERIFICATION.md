# Release verification — October 21, 2026

The camera-ready code/results release was checked with Python 3.12 on the local
macOS environment. No new model download, paid cloud invocation or GPU sweep was
performed during preparation.

- **188 offline tests passed; 22 network/slow tests deselected.** Three Modal
  warnings concern mocked supervisor functions executing locally, not cloud jobs.
- All **217** bundled reference artifacts pass compressed and raw SHA-256 checks.
- The master table recomputes **16 model/lens conditions on 13 checkpoints** from
  saved outputs. Its provenance records **85** input/code hashes. The published
  figure displays **11 distinct checkpoints** using the declared display policy.
- The public snapshot checks **15 published files**, all plotted statistics
  against the full table, the figure generator hash and the recorded input hashes.
  Freshly rebuilt tables and plotting statistics match the snapshot exactly.
- The release allowlist/secret scan passes, including decompressed records and
  historical Torch metadata. The optional remote entrypoint and citation file
  are included; manuscripts and review correspondence remain excluded.
- The mechanism, lambda and outcome-breakdown figures rebuild from restored
  saved inputs. Their axes use same-position/filtered terminology.
- Release/publishing helpers and the cross-model plot generator pass Ruff;
  `git diff --check` passes. This is not a claim that all historical exploratory
  scripts satisfy every current lint rule.

The extracted archive is checked with restoration, snapshot verification,
all offline tests and CPU regeneration of the master table and cross-model
figures. Source imports are redirected to the extracted checkout; the existing
Python environment supplies dependencies. This is an extracted-source check,
not a newly installed dependency environment or a fresh GPU experiment.

These checks establish integrity and offline reanalysis of the recorded outputs.
They do not establish fresh GPU equivalence, repeated-fit stability, functional
layer equivalence across models or Linux CI success before its run finishes.
Published-lens filtering remains approximate. The unsourced overview composite,
historical population differences and original environment limits are described
in [the historical audit](PAPER_AUDIT.md).

The default source/reference archive exceeds the conference supplement size
limit. It is a code-host artifact; it is not a ready-to-upload ACCV supplement.
The final paper PDF and OpenReview fields require their own submission checks.
