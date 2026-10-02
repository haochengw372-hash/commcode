# Upgrading the local 0.1 prototype

Version 0.2.0 adds a typed API in `commcode.system1` and the `commcode` command.
It preserves the earlier binary annotation entry points rather than changing the
meaning of `Code`, `Decision`, or `p_present`.

- Use `Book` and `Dimension` for original categorical labels, yes/no probabilities,
  or ordered numeric anchors. Supply `EvidencePacket` explicitly.
- Keep final reference labels outside runtime evidence; they may be used in a
  separate offline calibration/evaluation step.
- Start new output directories for the new journal format. Existing binary run
  records are not automatically migrated into typed System 1 records.
- An unresolved network dispatch requires an explicit decision about retrying,
  since it may already have incurred a provider charge.
- The typed HTTP runtime has no required third-party dependencies. The optional
  legacy `DeepSeekBackend` still uses the `llm` extra.
- POSIX locking currently limits the typed journal implementation to macOS/Linux.

The typed API is alpha. Keep a frozen copy of codebooks, model configuration, and
software version with each study. A successful interface test establishes runtime
behavior; benchmark quality and applicability require separate validation.
