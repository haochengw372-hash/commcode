import tomllib
import unittest
from pathlib import Path


class CitationMetadataTests(unittest.TestCase):
    def test_citation_and_license_are_consistent(self):
        root = Path(__file__).resolve().parents[1]
        first_line = (root / "README.md").read_text().splitlines()[0]
        self.assertTrue(first_line.startswith("> 使用请引用："))
        self.assertIn("Haocheng Wang", first_line)
        self.assertIn("不是额外的许可证限制", first_line)
        citation = (root / "CITATION.cff").read_text()
        self.assertIn("family-names: Wang", citation)
        self.assertIn("given-names: Haocheng", citation)
        metadata = tomllib.loads((root / "pyproject.toml").read_text())
        self.assertEqual(metadata["project"]["license"], "Apache-2.0")
        self.assertIn(f'version: {metadata["project"]["version"]}', citation)
        self.assertIn("CITATION.cff", (root / "MANIFEST.in").read_text())


if __name__ == "__main__":
    unittest.main()
