# Version 0.3: corpus execution

`encode_corpus` partitions all supplied units and questions into bounded physical
requests, executes them with limited concurrency, applies a frozen calibrator to
base predictions, and optionally replaces selected cells with an LLM judgment.
The single-packet `System1Encoder` and legacy binary APIs remain available.

The optional `native` profile uses source-declared response meanings when choosing
typed primitives. Original labels and output scales are preserved. Pass
`prompt_family="native"` or `--prompt native` explicitly; legacy defaults remain.

Distinguish `packet_budget` (canonical request bytes) from policy
`budget_fraction` (fraction of physical packets receiving an extra LLM call).
The default assistance fraction is zero. `max_units` is per packet; it does not
truncate the corpus. Evidence too large for a valid request receives a review
status instead of silent clipping.

Use a new output directory or explicit new cache epoch when changing a frozen
workflow. Keep old artifacts for comparison. Version 0.2's local distributions
are preserved separately; neither version numbering nor source publication
establishes a completed PyPI release.
