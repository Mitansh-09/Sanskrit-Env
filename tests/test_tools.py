"""Unit tests for ManuscriptToolkit."""

import json
import unittest
from pathlib import Path
from server.tools import ManuscriptToolkit


class TestManuscriptToolkit(unittest.TestCase):
    def setUp(self):
        self.toolkit = ManuscriptToolkit()
        data_path = Path(__file__).parent.parent / "data" / "task5_restoration.json"
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.episodes = data["episodes"]
        # rest_001 has glossary, sandhi, meter, commentary data
        self.ep1 = self.episodes[0]

    def test_lexicon_lookup_found(self):
        result = self.toolkit.lexicon_lookup("hantāram", self.ep1)
        self.assertTrue(result["found"])
        self.assertGreater(len(result["meanings"]), 0)

    def test_lexicon_lookup_not_found(self):
        result = self.toolkit.lexicon_lookup("nonexistent", self.ep1)
        self.assertFalse(result["found"])

    def test_sandhi_parser_found(self):
        result = self.toolkit.sandhi_parser("yaścainaṃ", self.ep1)
        self.assertTrue(result["found"])
        self.assertGreater(len(result["splits"]), 0)

    def test_sandhi_parser_not_found(self):
        result = self.toolkit.sandhi_parser("nonexistent", self.ep1)
        self.assertFalse(result["found"])

    def test_meter_checker_valid(self):
        result = self.toolkit.meter_checker("yaś ca enaṃ", self.ep1)
        self.assertTrue(result["found"])
        self.assertTrue(result["preserves_meter"])
        self.assertEqual(result["meter_name"], "anuṣṭubh")

    def test_meter_checker_invalid(self):
        result = self.toolkit.meter_checker("yaś cai naṃ", self.ep1)
        self.assertTrue(result["found"])
        self.assertFalse(result["preserves_meter"])

    def test_commentary_fetch_found(self):
        result = self.toolkit.commentary_fetch("hantāram", self.ep1)
        self.assertTrue(result["found"])
        self.assertIn("ātman", result["commentary"].lower())

    def test_commentary_fetch_not_found(self):
        result = self.toolkit.commentary_fetch("nonexistent", self.ep1)
        self.assertFalse(result["found"])

    def test_witness_compare_no_data(self):
        # rest_001 (beginner) has empty witness_data
        result = self.toolkit.witness_compare("2.19", self.ep1)
        self.assertFalse(result["found"])

    def test_referent_tracker_found(self):
        result = self.toolkit.referent_tracker("enaṃ", self.ep1)
        self.assertTrue(result["found"])
        self.assertGreater(len(result["antecedents"]), 0)

    def test_dispatch_unknown_tool(self):
        result = self.toolkit.dispatch("fake_tool", "input", self.ep1)
        self.assertIn("error", result)

    def test_dispatch_valid_tool(self):
        result = self.toolkit.dispatch("lexicon_lookup", "hantāram", self.ep1)
        self.assertTrue(result["found"])

    def test_catalog_returns_six_tools(self):
        catalog = ManuscriptToolkit.catalog()
        self.assertEqual(len(catalog), 6)
        names = {t["name"] for t in catalog}
        self.assertEqual(names, set(ManuscriptToolkit.TOOL_NAMES))


if __name__ == "__main__":
    unittest.main()
