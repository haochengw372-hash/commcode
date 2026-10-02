"""Verify public archive hashes and a fresh install for a published version."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import venv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", choices=("testpypi", "pypi"), required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    domain = "test.pypi.org" if args.index == "testpypi" else "pypi.org"
    url = f"https://{domain}/pypi/commcode/{args.version}/json"
    public = None
    for attempt in range(12):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                public = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            if exc.code != 404 or attempt == 11:
                raise
            time.sleep(10)
    assert public is not None
    hashes = {x["filename"]: x["digests"]["sha256"] for x in public["urls"]}
    checked = {}
    for path in sorted(args.dist.iterdir()):
        if path.suffix != ".whl" and not path.name.endswith(".tar.gz"):
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashes.get(path.name) != actual:
            raise ValueError(f"Published hash differs: {path.name}")
        checked[path.name] = actual
    if len(checked) != 2:
        raise ValueError("Expected exactly one verified wheel and one sdist")
    example = Path(__file__).resolve().parents[1] / "examples" / "offline_demo.py"
    with tempfile.TemporaryDirectory(prefix="commcode-public-install-") as folder:
        root = Path(folder)
        venv.create(root / "env", with_pip=True)
        python = root / "env" / "bin" / "python"
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--only-binary=:all:",
                "--index-url",
                f"https://{domain}/simple/",
                f"commcode=={args.version}",
            ],
            cwd=root,
            check=True,
        )
        installed = subprocess.check_output(
            [
                str(python),
                "-I",
                "-c",
                "import importlib.metadata, commcode, json; "
                "print(json.dumps({'version':importlib.metadata.version('commcode'),"
                "'path':commcode.__file__}))",
            ],
            cwd=root,
            text=True,
        )
        info = json.loads(installed)
        if info["version"] != args.version or str(root / "env") not in info["path"]:
            raise ValueError("Install did not resolve to the fresh environment")
        subprocess.run([str(python), "-I", str(example)], cwd=root, check=True)
        subprocess.run(
            [str(python), "-I", "-m", "commcode.system1", "--help"], cwd=root, check=True
        )
    args.output.write_text(
        json.dumps(
            {
                "index": args.index,
                "version": args.version,
                "public_hashes_verified": checked,
                "clean_install_and_offline_cli": "passed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
