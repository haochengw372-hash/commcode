# CommCode

CommCode is a Python library and command-line tool for coding a corpus with an
original research codebook. It compiles declared questions to typed AI judgments,
partitions the corpus without clipping evidence, and keeps resumable requests,
responses, model identity, usage and timing. Optional supervised calibration and
bounded LLM assistance operate on the original labels and scales.

Version **0.3.0** adds the full-corpus `encode_corpus` API. Python 3.11+ and
macOS/Linux are supported; the durable journal currently uses POSIX file locks.
The core runtime uses only the standard library. Apache-2.0 covers this software,
not third-party research data. No corpus, human-reference labels, media, credentials
or model weights are included.

## Installation and an offline example

```sh
python -m pip install .
commcode --help
python examples/offline_demo.py
```

The example uses an injected synthetic transport, makes no provider calls, and
checks all three primitive types, 1,000-unit corpus coverage, and zero-request
cache replay. It demonstrates software behavior rather than empirical accuracy.
For a public-index installation use `python -m pip install commcode==0.3.0` only
after that version appears on [PyPI](https://pypi.org/project/commcode/). GitHub
source availability and a passing local build do not imply PyPI publication.

## Code a corpus

Use [`examples/book.json`](examples/book.json) and
[`examples/units.json`](examples/units.json) as synthetic JSON templates. Preserve
original question wording, response meanings, exclusions and examples in your
book. Keep human-reference labels in a separate offline evaluation file.
Configure `TYPESAFE_API_KEY` securely in the process environment, then run:

```sh
commcode --codebook examples/book.json --units examples/units.json \
  --output run --provider jev --model jev-latest --prompt native \
  --packet-budget 180000 --cell-budget 256 --max-units 16 --workers 4 \
  --budget-fraction 0 --cache-epoch study-v1
```

This command makes real provider calls and may incur charges. The Jev endpoint is
`https://api.typesafe.ai/v1/systemone`. The package does not load `.env` files.
Aliases such as `jev-latest` do not pin model weights; retain returned model
identity and revalidate provider changes. A provider-supported fixed version can
be supplied with `--model`.

```python
import json
from pathlib import Path
from commcode.system1 import Book, Provider, encode_corpus

book = Book.from_dict(json.loads(Path("examples/book.json").read_text()))
units = json.loads(Path("examples/units.json").read_text())
result = encode_corpus(
    units, book, output_dir="run",
    jev_provider=Provider("jev", model="jev-latest"),
    prompt_family="native",
    packet_budget=180000, cell_budget=256, max_units=16, workers=4,
    policy={"budget_fraction": 0.0, "cell_fraction": 1.0},
    cache_epoch="study-v1",
)
```

`max_units` limits each request packet, never the total corpus. Every requested
unit/dimension must be accounted for. Oversized source evidence is reported for
review rather than silently truncated. Identical supplied context can be shared
within a packet; unrelated units do not acquire each other's context. Raw video
extraction, audio transcription and verification of upstream observation coverage
remain separate from this textual backend.

## Typed questions and original coding semantics

- **Choice** retains original category IDs and their probabilities.
- **Noul** estimates the probability of the specified yes/no event. A declared
  `rater_fraction` estimand asks about a sampled rater's response; it is not
  automatically interchangeable with the model's own confidence.
- **Score** uses ordered source anchors and maps its native index to the declared
  `output_range`. Report results on that original scale.

The explicit `native` profile projects an unambiguous source Yes/No Choice to
Noul and then back to the same original category IDs. Opaque numeric IDs require
an explicit original response mapping; the compiler does not guess their meanings.
Existing population-fraction Noul and Score questions keep their primitive and
scale. Conditional questions retain a supplied source parent binding. Complete
original codebook text is still provided.

`native` is the current research-selected profile, not a universally proven
winner. Pass it explicitly: lower-level API defaults and the CLI's legacy
`compact` default are retained for compatibility. Other available profiles are
`verbose`, `compact`, `anchored`, `boundary`, and `grounded`. Joint A/B coding and
explanation-based research candidates are not automatically enabled.

## Calibration and optional assistance

Fit a calibration artifact offline using final reference labels and disjoint
source groups; see [calibration.md](docs/calibration.md). Supply the frozen
artifact as `calibrator=...` in Python or `--calibration-model model.json` in the
CLI. Identity is a candidate. Original labels/ranges are preserved, provider
`confidence` remains distinct from `calibrated_pmax`, and the fitted transform
and input provenance are recorded. Threshold utility fitting requires an explicit
choice during offline fitting.

The order is **base Jev judgments → base-only calibration → risk selection →
optional targeted LLM replacements**. LLM replacement labels are not processed
through a Jev probability calibrator. Runtime corpus coding accepts no reference
labels. The package removes known reference metadata, but callers must still
ensure that free-text evidence itself contains no answer leakage.

The default assistance budget is zero. To enable it, explicitly supply an
`llm_provider=Provider("llm", model="your-supported-model")` and a nonzero policy.
Configure `OPENAI_API_KEY` and `OPENAI_BASE_URL`; the CLI offers `--llm-model` and
`--llm-endpoint`. Never put credentials in URLs or command arguments.

These similarly named limits have different units:

| Setting | Meaning |
| --- | --- |
| `packet_budget` / `--packet-budget` | Maximum canonical request **bytes**, e.g. 180000; not tokens or an escalation fraction |
| `cell_budget` / `--cell-budget` | Maximum coding questions in a base packet |
| `max_units` / `--max-units` | Maximum source units per packet |
| Policy `budget_fraction` / `--budget-fraction` | Fraction of actual request packets eligible for an additional LLM request, e.g. 0.25 |
| Policy `cell_fraction` / `--cell-fraction` | Fraction of cells selected within an upgraded packet |

Fractional limits use floor rounding and can yield zero requests. They cap request
counts, not money. Actual bytes, tokens, usage, additional calls and elapsed time
must be measured. No accuracy, timing or cost superiority is implied by enabling
calibration or assistance; current research selection precedes final independent
confirmation and paired end-to-end timing.

## Auditing, resume and compatibility

The run contains a workflow manifest, result and durable provider journals. A
signature covers actual requests, source inputs, endpoints/models, calibration,
policy and cache epoch. Received raw responses are stored before parsing.
Supported lossless output normalization is marked while retaining raw output.
An unresolved dispatch may already have been charged and is not automatically
resent. A new retry epoch is an explicit action. These records contain evidence
and provider output; keep them private when your corpus is private.

`System1Encoder` remains available for one explicit packet. Its `postprocess`
callback applies to base judgments before selection, not to LLM replacements.
The earlier binary API (`Code`, `Unit`, `annotate`, loaders, `score_decisions`)
remains available without reinterpreting `present`/`absent`. Its optional legacy
LLM adapter uses `python -m pip install 'commcode[llm]'`.

```sh
python -m pip install '.[dev]'
python -m pytest
python -m ruff check src tests examples scripts
```

See [LICENSE](LICENSE), [NOTICE](NOTICE), and [release instructions](docs/releasing.md).
Third-party dataset licenses continue to apply when users obtain data separately.
