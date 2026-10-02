# CommCode

CommCode is a Python library and command-line tool for coding communication material
with an original research codebook. It sends declared evidence to a decision model
or a compatible language-model endpoint, validates typed answers, and keeps a
resumable record of requests, responses, model identity, usage and timing.

Version **0.2.0** introduces `commcode.system1` alongside the existing binary API.
The earlier `0.1.0` was a local prototype. Public availability must be checked on
[PyPI](https://pypi.org/project/commcode/); a version number in this checkout is not
proof of publication. The current journal implementation supports macOS and Linux
(POSIX file locks), with Python 3.11 or newer.

## Install

From a release checkout, install with:

```sh
python -m pip install .
commcode --help
```

For a published version, use `python -m pip install commcode==0.2.0` after confirming
that version exists on PyPI. The typed System 1 runtime uses the Python standard
library and has no required third-party dependencies. The optional legacy LLM
adapter is installed with `python -m pip install 'commcode[llm]'`.

## Try the synthetic example offline

```sh
python examples/offline_demo.py
```

This example injects a synthetic response, exercises output validation and resume,
and makes no provider call. It demonstrates the interface, not coding accuracy.
All examples in this repository are synthetic.

## Code real material

Supply the original codebook and evidence as JSON, following
[`examples/book.json`](examples/book.json) and
[`examples/units.json`](examples/units.json). Configure `TYPESAFE_API_KEY` securely
in your process environment, then run:

```sh
commcode --codebook examples/book.json --units examples/units.json \
  --output run --provider jev --model jev-latest
```

That command makes a real TypeSafe call and may incur provider charges. The default
Jev endpoint is `https://api.typesafe.ai/v1/systemone`. `jev-latest` is an alias;
recorded returned model identity does not guarantee immutable provider weights.
Use a provider-supported fixed version when available and revalidate changed models.
The package does not automatically load `.env` files.

For a compatible LLM endpoint, configure `OPENAI_API_KEY` and `OPENAI_BASE_URL`, use
`--provider llm`, and set its supported `--model`. `--endpoint` can explicitly select
a full endpoint URL. Never put credentials in endpoint URLs or command arguments.

```python
from commcode.system1 import Book, EvidencePacket, Provider, System1Encoder

book = Book.from_dict({
    "book_id": "synthetic-v1",
    "original_text": "Use request for a concrete request, otherwise other.",
    "dimensions": [{
        "id": "request", "question": "Is there a concrete request?",
        "kind": "choice", "labels": ["request", "other"],
    }],
})
packet = EvidencePacket.from_dict({"units": [{
    "unit_id": "example-1", "text": "Please add a search box.", "context": {},
}]})
result = System1Encoder(Provider("jev")).encode(packet, book, output_dir="run")
```

## Typed questions and evidence

`Dimension` supports `choice` (original categories and their probabilities), `noul`
(a probability for the stated yes/no event), and `score` (ordered anchors mapped to
an explicitly declared original range). A model probability is not automatically a
population response fraction; declare the intended estimand and validate it with
appropriate reference data. Missing or invalid outputs require review and are not
silently converted to an absent/negative category.

`Book` retains original text and optional source spans. `EvidencePacket` keeps
complete supplied unit text and declared context. Runtime inputs and evaluation
references should be separate. The input filter removes known reference metadata,
but users must still ensure that free-text evidence contains no answer labels.
The text backend accepts textual multimodal observations; this release does not
extract frames, transcribe audio, or infer that observations cover an entire video.

`System1Encoder` optionally accepts explicit `selector` and `postprocess` callbacks.
Selection can request an additional compatible LLM judgment for named targets;
its cost is additional. No unvalidated confidence threshold or data-specific
routing policy is silently enabled. Calibration uses final reference labels offline;
the release does not claim a universal accuracy, speed, or cost advantage.

## Journal and resume

Each output directory stores a journal and `result.json`. A request signature
includes endpoint, model request, codebook/evidence and cache epoch. A durable
claim is written before dispatch; received raw responses are saved before parsing.
An unresolved dispatch may have been charged and is not automatically resent.
Parsing errors retain the response. Keep these local artifacts private when your
corpus is private: they contain input evidence and model output.

The original binary interface (`Code`, `Unit`, `annotate`, loaders and
`score_decisions`) remains available. Its `present`/`absent` decisions are not
reinterpreted as the new typed categories. Migrate multiclass or numeric work to
`commcode.system1` explicitly.

## Development and license

```sh
python -m pip install '.[dev]'
python -m pytest
python -m ruff check src tests examples scripts
```

Software is licensed under Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
No research corpora, human labels, videos, credentials or model weights are included.
Their original terms remain applicable when obtained separately. In particular,
this software license does not relicense third-party dataset content.
See [release instructions](docs/releasing.md) for artifact auditing, clean installs,
and staged TestPyPI/PyPI Trusted Publishing.
