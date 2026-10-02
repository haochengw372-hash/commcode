# Public aggregate research results

[Available-evidence snapshot, 2026-10-03](system1_available_20261003/README.md)
contains the approved aggregate tables and all 193 figures in PNG, PDF and SVG,
with their 193 numeric source CSVs. See the [file index](system1_available_20261003/index.json),
[privacy/export audit](system1_available_20261003/audit.json), and
[figure catalog](system1_available_20261003/figure_catalog.csv).

This is an incomplete-research checkpoint. The full LLM confirmation comparator
and formal paired end-to-end timing remain pending. It does not establish two
performance advantages, a universal optimum, or a published PyPI package.
The pending paper is not presented here as a finished manuscript.

Only derived aggregates and authored charts are published. Raw inputs, unit-level
predictions/reference rows, private full texts and provider response journals
remain local. The software's Apache-2.0 license does not relicense third-party
corpora or codebooks.

The [exporter](tools/export_research_results.py) and its
[explicit allowlist](tools/research_export_allowlist.json) reproduce the public
snapshot from the corresponding private research-run directory:

```sh
python research_results/tools/export_research_results.py --self-test
python research_results/tools/export_research_results.py \
  --source /path/to/private/research-run \
  --destination /path/to/new-public-snapshot
```

The exporter requires `pdftotext` for a privacy scan of rendered PDF text.
The source directory must contain the completed analysis, plots and frozen policy
artifacts. Export does not download third-party datasets or execute model calls.
Use a new destination for a revised snapshot; existing snapshots are preserved.
