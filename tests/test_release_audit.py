import importlib.util
from pathlib import Path


def load_auditor():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_release.py"
    spec = importlib.util.spec_from_file_location("audit_release", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_distribution_allows_exact_root_citation_file():
    auditor = load_auditor()
    assert auditor.allowed("CITATION.cff", wheel=False)
    assert not auditor.allowed("other.cff", wheel=False)
    assert not auditor.allowed("docs/CITATION.cff", wheel=False)
    assert not auditor.allowed("../CITATION.cff", wheel=False)
    assert not auditor.allowed("CITATION.cff", wheel=True)
