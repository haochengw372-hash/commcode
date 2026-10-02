"""Export an explicit aggregate-only public research snapshot; never package raw inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

BAD_KEYS = {
    "unit_id",
    "unit_ids",
    "group_id",
    "group_ids",
    "batch_id",
    "record_path",
    "journal_path",
    "input_text",
    "text",
    "reference",
    "original_reference",
    "probabilities",
    "request",
    "response",
    "raw_response",
    "caption",
    "transcript",
    "api_key",
    "authorization",
    "selection_path",
}
PATTERNS = [
    re.compile(
        rb"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}|pypi-[A-Za-z0-9_-]{30,})\b"
    ),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"/" + rb"Users/[^/\s]+/"),
    re.compile(rb"/" + rb"home/[^/\s]+/"),
]
GROUPS = {
    "attempts.csv": (
        [
            "epoch",
            "dataset",
            "split",
            "stage_directory",
            "status",
            "http_status",
            "model_alias",
            "returned_model",
            "billing_status",
        ],
        [
            "api_elapsed_seconds",
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "cold_cost_usd_low",
            "cold_cost_usd_high",
            "warm_cost_usd_low",
            "warm_cost_usd_high",
        ],
    ),
    "costs.csv": (
        [
            "epoch",
            "dataset",
            "split",
            "method",
            "stage",
            "execution_type",
            "effective_family",
            "kind",
            "cache_hit",
            "status",
            "billing_status",
            "returned_model",
        ],
        [
            "api_elapsed_seconds",
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "cold_cost_usd_low",
            "cold_cost_usd_high",
            "warm_cost_usd_low",
            "warm_cost_usd_high",
        ],
    ),
    "registry.csv": (
        [
            "epoch",
            "dataset",
            "split",
            "method",
            "status",
            "execution_type",
            "current_book_match",
            "family",
            "effective_families",
        ],
        ["n_units_expected", "n_cells_returned", "workflow_wall_seconds"],
    ),
    "excluded_references.csv": (
        ["epoch", "dataset", "split", "method", "dimension_id", "reason"],
        [],
    ),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def audit_bytes(data, name):
    if any(pattern.search(data) for pattern in PATTERNS):
        raise ValueError(f"Private path or credential pattern in {name}")


def audit_json(value, path="root"):
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in BAD_KEYS:
                raise ValueError(f"Granular/private JSON field at {path}.{key}")
            audit_json(child, f"{path}.{key}")
    elif isinstance(value, list):
        for child in value:
            audit_json(child, path)
    elif isinstance(value, str):
        audit_bytes(value.encode(), path)


def csv_bytes(fields, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def read_csv(data):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    return reader.fieldnames or [], list(reader)


def safe_csv(data, allowed, name):
    fields, rows = read_csv(data)
    if not set(fields) <= set(allowed) or set(fields) & BAD_KEYS:
        raise ValueError(
            f"Unapproved aggregate CSV fields in {name}: {sorted(set(fields) - set(allowed))}"
        )
    audit_bytes(data, name)
    return len(rows)


def group_csv(data, name):
    fields, rows = read_csv(data)
    dimensions, measures = GROUPS[name]
    dimensions = [key for key in dimensions if key in fields]
    measures = [key for key in measures if key in fields]
    grouped = defaultdict(lambda: {"record_count": 0})
    for row in rows:
        key = tuple(row.get(k, "") for k in dimensions)
        target = grouped[key]
        target["record_count"] += 1
        for measure in measures:
            value = row.get(measure, "")
            if value not in ("", "None", "null"):
                try:
                    number = float(value)
                except ValueError as exc:
                    raise ValueError("Non-numeric aggregate measure") from exc
                target["sum_recorded_" + measure] = (
                    target.get("sum_recorded_" + measure, 0) + number
                )
    result = [
        {**dict(zip(dimensions, key)), **value}
        for key, value in sorted(grouped.items())
    ]
    headers = (
        dimensions + ["record_count"] + ["sum_recorded_" + key for key in measures]
    )
    for row in result:
        for key in headers:
            row.setdefault(key, "")
    return csv_bytes(headers, result), len(result)


def export(source, destination, allowlist):
    source = source.resolve()
    if destination.exists():
        raise ValueError("Snapshot destination already exists; use a new version")
    expected = json.loads(allowlist.read_text())
    index, figures = [], []
    candidate_sources = [
        source / "analysis" / name
        for name in expected["direct_csv"]
        + expected["direct_json"]
        + list(GROUPS)
        + ["routing.csv"]
    ]
    candidate_sources += [
        p
        for p in (source / "plots").iterdir()
        if p.is_file() and p.suffix in expected["plot_extensions"]
    ]
    candidate_sources += [
        source / "analysis/figure_catalog.csv",
        source / "selection/final_policy_v1.json",
    ]
    all_sources = {str(p): sha(p.read_bytes()) for p in candidate_sources}
    with tempfile.TemporaryDirectory(
        prefix="commcode-public-export-", dir=destination.parent
    ) as temp:
        stage = Path(temp)

        def write(
            relative, data, source_file=None, transformation="byte-identical", rows=None
        ):
            audit_bytes(data, relative)
            path = stage / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            entry = {
                "public_path": relative,
                "public_sha256": sha(data),
                "bytes": len(data),
                "transformation": transformation,
            }
            if source_file is not None:
                original = source_file.read_bytes()
                if sha(original) != all_sources[str(source_file)]:
                    raise RuntimeError("Source changed during snapshot preparation")
                entry.update(
                    source_path=str(source_file.relative_to(source)),
                    source_sha256=all_sources[str(source_file)],
                )
            if rows is not None:
                entry["rows"] = rows
            index.append(entry)

        for name in expected["direct_csv"]:
            original = source / "analysis" / name
            data = original.read_bytes()
            n = safe_csv(data, expected["aggregate_csv_fields"], name)
            write("tables/" + name, data, original, rows=n)
        for name in expected["direct_json"]:
            original = source / "analysis" / name
            value = json.loads(original.read_text())
            transformation = "JSON reserialized; no content fields removed"
            if name == "available_evidence_status.json":
                value["frozen_policy_path"] = "frozen_policy_summary.json"
                transformation = (
                    "Repointed frozen_policy_path to public summary beside this file"
                )
            audit_json(value)
            write(
                "metadata/" + name,
                json.dumps(value, ensure_ascii=False, indent=2).encode(),
                original,
                transformation,
            )
        policy_file = source / "selection/final_policy_v1.json"
        policy = json.loads(policy_file.read_text())
        policy_keys = [
            "created_at_utc",
            "status",
            "architecture",
            "profile",
            "prompt_family",
            "question_compiler_version",
            "calibration",
            "models",
            "optional_llm_route",
            "selection_reason",
            "confirmation_exposure_note",
            "timing",
            "success_dimensions",
        ]
        public_policy = {key: policy[key] for key in policy_keys if key in policy}
        public_policy["default_physical_packet_fraction_budget"] = policy.get(
            "default_packet_budget"
        )
        public_policy["default_within_packet_cell_fraction_budget"] = policy.get(
            "default_cell_budget"
        )
        audit_json(public_policy)
        write(
            "metadata/frozen_policy_summary.json",
            json.dumps(public_policy, indent=2).encode(),
            policy_file,
            "Whitelisted methodological summary; private calibration/input artifacts and unit IDs omitted",
        )
        catalog_file = source / "analysis/figure_catalog.csv"
        catalog_fields, catalog_rows = read_csv(catalog_file.read_bytes())
        if set(catalog_fields) != {
            "figure",
            "png",
            "pdf",
            "svg",
            "data_csv",
            "png_sha256",
        }:
            raise ValueError("Unexpected source figure catalog schema")
        for row in catalog_rows:
            for field in ("png", "pdf", "svg", "data_csv"):
                original_path = Path(row[field])
                relative_path = original_path.resolve().relative_to(source)
                if relative_path.parts[0] != "plots":
                    raise ValueError("Catalog link outside approved plot directory")
                row[field] = str(relative_path)
            if sha((source / row["png"]).read_bytes()) != row["png_sha256"]:
                raise ValueError("Source catalog figure hash mismatch")
        write(
            "figure_catalog.csv",
            csv_bytes(catalog_fields, catalog_rows),
            catalog_file,
            "All original catalog fields retained; absolute paths rewritten to snapshot-relative published plots",
            len(catalog_rows),
        )
        for name in GROUPS:
            original = source / "analysis" / name
            data, n = group_csv(original.read_bytes(), name)
            write(
                "tables/" + name.replace(".csv", "_aggregate.csv"),
                data,
                original,
                "Grouped counts/sums; identifiers, paths and individual records omitted",
                n,
            )
        routing = source / "analysis/routing.csv"
        fields, rows = read_csv(routing.read_bytes())
        fields = [key for key in fields if key != "selection_path"]
        if set(fields) & BAD_KEYS:
            raise ValueError("Unexpected private routing fields")
        write(
            "tables/routing.csv",
            csv_bytes(
                [
                    {
                        "packet_budget": "physical_packet_fraction_budget",
                        "cell_budget": "within_packet_cell_fraction_budget",
                    }.get(k, k)
                    for k in fields
                ],
                [
                    {
                        (
                            {
                                "packet_budget": "physical_packet_fraction_budget",
                                "cell_budget": "within_packet_cell_fraction_budget",
                            }.get(k, k)
                        ): r[k]
                        for k in fields
                    }
                    for r in rows
                ],
            ),
            routing,
            "Removed private selection_path; renamed fraction budgets to avoid confusion with byte limits",
            len(rows),
        )
        plots = sorted((source / "plots").iterdir())
        for original in plots:
            if (
                not original.is_file()
                or original.suffix not in expected["plot_extensions"]
            ):
                continue
            data = original.read_bytes()
            if original.suffix == ".csv":
                safe_csv(data, expected["aggregate_csv_fields"], original.name)
            if original.suffix == ".pdf":
                if not shutil.which("pdftotext"):
                    raise RuntimeError(
                        "pdftotext is required for PDF text privacy audit"
                    )
                text = subprocess.check_output(["pdftotext", str(original), "-"])
                audit_bytes(text, original.name)
            write("plots/" + original.name, data, original)
            if original.suffix in (".pdf", ".svg", ".png"):
                stem = original.stem
                csv_source = source / "plots" / (stem + "_data.csv")
                if not csv_source.exists() and stem.endswith("_tradeoffs"):
                    csv_source = (
                        source
                        / "plots"
                        / (stem.removesuffix("_tradeoffs") + "_comparison_data.csv")
                    )
                if not csv_source.exists():
                    raise ValueError(
                        f"Figure lacks an audited aggregate CSV companion: {original.name}"
                    )
                figures.append(
                    {
                        "figure_path": "plots/" + original.name,
                        "data_path": "plots/" + csv_source.name,
                        "data_mapping": "companion aggregate CSV",
                    }
                )
        write(
            "figure_data_index.csv",
            csv_bytes(["figure_path", "data_path", "data_mapping"], figures),
            transformation="Generated public figure-to-aggregate-data index",
        )
        write(
            "export_allowlist.json",
            allowlist.read_bytes(),
            transformation="Explicit public export allowlist",
        )
        readme = """# Aggregate research snapshot

This snapshot includes the complete approved aggregate analysis tables, grouped
failure/usage/coverage accounting, and all available PNG/SVG/PDF figures with
aggregate plotting data. It contains no unit-level predictions, reference rows,
source texts, transcripts, media, provider responses, credentials or private PDFs.

Research remains incomplete: the full LLM confirmation comparator and formal
paired end-to-end timing are not complete. Do not infer two proven advantages,
universal superiority, or PyPI publication. Negative and partial results remain
visible. The pending manuscript is not published here as a final paper.

`index.json` records each source artifact's logical path relative to the private
research-run root, its SHA-256, and its public copy's repository-relative path and
SHA-256. Source paths are provenance identifiers, not promises that private files
are present in this repository. All public links are relative to this snapshot.
`figure_catalog.csv` links figures to approved aggregate data. No absolute local
paths are retained.

The attempt, cost, registry and exclusion tables are grouped before export.
`record_count` counts source table records. `sum_recorded_*` adds recorded values;
it is not necessarily unique expenditure, since methods can reuse journals.
Use summary tables' accounting and research_usage.json for their stated distinct
scopes. Cumulative API durations do not establish wall-clock throughput.

Software is Apache-2.0. This does not relicense third-party corpora or codebooks;
obtain them separately under their original licenses. This snapshot contains only
derived aggregates and authored charts. Raw reproducibility records remain local.
"""
        write(
            "README.md",
            readme.encode(),
            transformation="Authored scope and accounting notice",
        )
        # Check stability after copying to reject a mixed snapshot from a running analysis.
        initial_plots = {
            path for path in all_sources if Path(path).parent == source / "plots"
        }
        final_plots = {
            str(p)
            for p in (source / "plots").iterdir()
            if p.is_file() and p.suffix in expected["plot_extensions"]
        }
        if initial_plots != final_plots:
            raise RuntimeError("Plot inventory changed during export")
        if any(
            sha(Path(path).read_bytes()) != value for path, value in all_sources.items()
        ):
            raise RuntimeError(
                "Analysis changed during export; no public snapshot written"
            )
        audit = {
            "status": "passed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "files_audited": len(index),
            "figure_files": len(figures),
            "source_hashes_rechecked": True,
            "unit_level_outputs_exported": False,
            "secret_and_private_path_patterns": "passed",
            "pdf_extracted_text_audit": "passed",
            "csv_schema_allowlist": "passed",
            "excluded_original_tables": expected["never_copy"],
        }
        (stage / "audit.json").write_text(json.dumps(audit, indent=2))
        (stage / "index.json").write_text(
            json.dumps(
                {
                    "snapshot_status": "incomplete-research-aggregate-export",
                    "source_root_role": "private system1 research run; logical source paths only",
                    "files": index,
                },
                indent=2,
            )
        )
        shutil.copytree(stage, destination)
    return audit


def self_test():
    assert safe_csv(b"dataset,accuracy\na,0.5\n", ["dataset", "accuracy"], "safe") == 1
    for payload in (b"unit_id,accuracy\nu1,1\n", b"dataset,text\na,private source\n"):
        try:
            safe_csv(payload, ["dataset", "accuracy"], "forbidden")
        except ValueError:
            pass
        else:
            raise AssertionError("Private schema accepted")
    for value in (
        {"unit_id": "x"},
        {"nested": {"reference": "x"}},
        {"key": "sk-" + "a" * 30},
    ):
        try:
            audit_json(value)
        except ValueError:
            pass
        else:
            raise AssertionError("Private JSON accepted")
    data, n = group_csv(
        b"epoch,dataset,status,unit_id,input_tokens\nv1,a,error,private1,12\nv1,a,error,private2,3\n",
        "attempts.csv",
    )
    assert n == 1 and b"private" not in data and b"15.0" in data
    print("Exporter negative privacy/schema tests passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=Path(__file__).with_name("research_export_allowlist.json"),
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.source or not args.destination:
        parser.error("--source and --destination are required")
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(export(args.source, args.destination, args.allowlist), indent=2))


if __name__ == "__main__":
    main()
