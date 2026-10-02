"""Run an original codebook against local, reference-free evidence packets."""

import argparse
import json
from pathlib import Path

from . import Book, EvidencePacket, Provider, System1Encoder
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
        "--prompt", choices=("compact", "verbose", "anchored", "boundary"), default="compact"
    )
    parser.add_argument("--cache-epoch", default="v1")
    args = parser.parse_args()
    try:
        book = Book.from_dict(json.loads(Path(args.codebook).read_text(encoding="utf-8")))
        packet = EvidencePacket.from_dict(json.loads(Path(args.units).read_text(encoding="utf-8")))
        result = System1Encoder(Provider(args.provider, args.model, args.endpoint)).encode(
            packet,
            book,
            output_dir=args.output,
            prompt_family=args.prompt,
            cache_epoch=args.cache_epoch,
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
                "cache_hit": result["record"]["cache_hit"],
            }
        )
    )


if __name__ == "__main__":
    main()
