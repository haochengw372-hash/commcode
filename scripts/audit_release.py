"""Audit wheel/sdist members before publication; never display matching secrets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT_FILES = {
    "README.md",
    "LICENSE",
    "NOTICE",
    "CITATION.cff",
    "pyproject.toml",
    "MANIFEST.in",
    "PKG-INFO",
    "setup.cfg",
}
FORBIDDEN_PARTS = {
    ".env",
    ".git",
    ".venv",
    ".venv-release",
    "__pycache__",
    "datasets",
    "media",
    "weights",
    "cache",
    "output",
    "journal",
    "runs",
}
PATTERNS = [
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(
        rb"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}|pypi-[A-Za-z0-9_-]{30,})\b"
    ),
    re.compile(rb"/" + rb"Users/[^/\s]+/"),
]


def allowed(name: str, wheel: bool) -> bool:
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or FORBIDDEN_PARTS.intersection(p.parts):
        return False
    if wheel:
        return (p.parts[0] == "commcode" and p.suffix == ".py") or (
            p.parts[0].startswith("commcode-")
            and p.parts[0].endswith(".dist-info")
            and (
                len(p.parts) == 2
                and p.name in {"METADATA", "WHEEL", "RECORD", "entry_points.txt", "top_level.txt"}
                or len(p.parts) == 3
                and p.parts[1] == "licenses"
                and p.name in {"LICENSE", "NOTICE"}
            )
        )
    if len(p.parts) == 1:
        return name in ROOT_FILES
    if p.parts[:2] == ("src", "commcode"):
        return p.suffix == ".py"
    if p.parts[:2] == ("src", "commcode.egg-info"):
        return p.name in {
            "PKG-INFO",
            "SOURCES.txt",
            "dependency_links.txt",
            "entry_points.txt",
            "requires.txt",
            "top_level.txt",
        }
    return p.parts[0] in {"tests", "docs", "examples", "scripts"} and p.suffix in {
        ".py",
        ".md",
        ".json",
    }


def audit(path: Path) -> dict:
    wheel = path.suffix == ".whl"
    if wheel:
        with zipfile.ZipFile(path) as archive:
            members = [(x.filename, archive.read(x)) for x in archive.infolist() if not x.is_dir()]
    else:
        with tarfile.open(path, "r:gz") as archive:
            members = []
            for member in archive.getmembers():
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError("Distribution contains a link or special file")
                name = str(
                    PurePosixPath(member.name).relative_to(PurePosixPath(member.name).parts[0])
                )
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("Unreadable archive member")
                members.append((name, stream.read()))
    failures = []
    for name, content in members:
        if not allowed(name, wheel):
            failures.append({"file": name, "reason": "not_allowlisted"})
        if any(pattern.search(content) for pattern in PATTERNS):
            failures.append({"file": name, "reason": "potential_secret_or_local_path"})
    if failures:
        raise ValueError(json.dumps(failures))
    return {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "members": sorted(name for name, _ in members),
        "status": "passed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"archives": [audit(path) for path in args.archives]}
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
