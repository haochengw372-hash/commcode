"""Run an original codebook against local, reference-free evidence packets."""

import argparse
import json
from pathlib import Path

from . import Book, EvidencePacket, Provider, encode_corpus
from .models import ValidationError
from .providers import DispatchUncertain, ProviderRejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--units", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provider", choices=("jev", "llm"), default="jev")
    parser.add_argument("--model")
    parser.add_argument("--endpoint")
    parser.add_argument(
        "--prompt",
        choices=("compact", "verbose", "anchored", "boundary", "grounded", "native"),
        default="compact",
    )
    parser.add_argument("--cache-epoch", default="v1")
    parser.add_argument("--calibration-model")
    parser.add_argument(
        "--packet-budget",
        type=int,
        default=180000,
        help="Maximum canonical bytes per physical request; source is never clipped",
    )
    parser.add_argument("--cell-budget", type=int, default=256)
    parser.add_argument(
        "--max-units",
        type=int,
        default=16,
        help="Maximum units per packet, not a corpus truncation",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--budget-fraction", type=float, default=0.0)
    parser.add_argument("--cell-fraction", type=float, default=1.0)
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-endpoint")
    args = parser.parse_args()
    try:
        book = Book.from_dict(json.loads(Path(args.codebook).read_text(encoding="utf-8")))
        packet = EvidencePacket.from_dict(json.loads(Path(args.units).read_text(encoding="utf-8")))
        calibration = (
            json.loads(Path(args.calibration_model).read_text(encoding="utf-8"))
            if args.calibration_model
            else None
        )
        llm = Provider("llm", args.llm_model, args.llm_endpoint) if args.budget_fraction else None
        result = encode_corpus(
            packet,
            book,
            output_dir=args.output,
            jev_provider=Provider(args.provider, args.model, args.endpoint),
            llm_provider=llm,
            calibrator=calibration,
            prompt_family=args.prompt,
            cache_epoch=args.cache_epoch,
            packet_budget=args.packet_budget,
            cell_budget=args.cell_budget,
            max_units=args.max_units,
            workers=args.workers,
            policy={"budget_fraction": args.budget_fraction, "cell_fraction": args.cell_fraction},
        )
    except (
        ValidationError,
        DispatchUncertain,
        ProviderRejected,
        OSError,
        KeyError,
        ValueError,
    ) as exc:
        # Exception type only: input paths and provider errors may include private details.
        parser.exit(1, f"Coding requires review: {type(exc).__name__}\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "cells": len(result["predictions"]),
                "usage": result["usage"],
                "wall_seconds": result["wall_seconds"],
            }
        )
    )
    if result["status"] != "completed":
        parser.exit(1, "Coding requires review; inspect result.json and the raw journal.\n")


if __name__ == "__main__":
    main()
