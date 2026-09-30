# Release and anonymous-review packaging

The release includes research code, frozen evaluation definitions, checksummed
reference artifacts, documentation and offline tests. Local manuscripts,
paper drafts, downloaded PDFs, notes, logs, credentials, caches and generated
results are excluded by `.gitignore` and by the archive's explicit inclusion
policy. These exclusions preserve local working files rather than deleting
research history. Only the 49 curated artifacts in `data/reference/` belong in
Git; the rest of the raw result workspace does not.

```bash
python scripts/release.py audit
python scripts/release.py verify
python scripts/release.py archive
```

`audit` also rejects unexpected tracked paths. For a private final check, pass
identifying strings as repeatable `--deny` arguments; do not put personal names
or account handles in a committed checklist. The scanner decompresses reference
files and examines Torch metadata as well as source text. It is a deterministic
check for known strings and common path/token patterns, not a guarantee against
all indirect identification.

The ZIP has a neutral root directory, fixed timestamps, SHA256SUMS and no Git
metadata. Extract it into a fresh directory, install dependencies, restore data,
and run tests/reproduction there before distribution. Anyone hosting a new
review repository should initialize new history with an anonymous author and
avoid an identifying account or link. Existing public history, forks, ownership
and prior publication cannot be anonymized by changing working-tree files.

Original project code is released under MIT, as selected by the rights holder.
Third-party attributions and terms remain in THIRD_PARTY_NOTICES.md. This
preparation does not rewrite public history or delete local working materials.
