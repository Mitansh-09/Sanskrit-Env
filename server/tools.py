"""
ManuscriptToolkit — six deterministic philological tools for Task 5.

All tools read from pre-annotated episode JSON data.
No external API calls. Pure functions.
"""

from typing import Any, Dict, List, Optional


class ManuscriptToolkit:
    """Six deterministic tools for manuscript restoration."""

    TOOL_NAMES = [
        "lexicon_lookup",
        "sandhi_parser",
        "meter_checker",
        "commentary_fetch",
        "witness_compare",
        "referent_tracker",
    ]

    def dispatch(
        self,
        tool_name: str,
        tool_input: str,
        episode: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Dispatch a tool call and return the result."""
        if tool_name not in self.TOOL_NAMES:
            return {"error": f"Unknown tool: {tool_name}", "available": self.TOOL_NAMES}

        handler = getattr(self, tool_name)
        return handler(tool_input, episode)

    def lexicon_lookup(self, term: str, episode: Dict[str, Any]) -> Dict[str, Any]:
        """Look up domain-specific meanings of a Sanskrit term."""
        glossary = episode.get("glossary_data", {})
        term_clean = term.strip()

        # Try exact match first, then case-insensitive
        entries = glossary.get(term_clean)
        if entries is None:
            for key, val in glossary.items():
                if key.lower() == term_clean.lower():
                    entries = val
                    break

        if entries is None:
            return {
                "term": term_clean,
                "found": False,
                "meanings": [],
                "note": f"Term '{term_clean}' not found in episode glossary.",
            }

        # Check if term appears in passage
        passage = episode.get("passage_iast", "")
        in_passage = term_clean.lower() in passage.lower()

        return {
            "term": term_clean,
            "found": True,
            "in_passage": in_passage,
            "meanings": entries,
        }

    def sandhi_parser(self, compound: str, episode: Dict[str, Any]) -> Dict[str, Any]:
        """Return phonologically valid splits of a compound."""
        sandhi_data = episode.get("sandhi_data", {})
        compound_clean = compound.strip()

        entries = sandhi_data.get(compound_clean)
        if entries is None:
            for key, val in sandhi_data.items():
                if key.lower() == compound_clean.lower():
                    entries = val
                    break

        if entries is None:
            return {
                "compound": compound_clean,
                "found": False,
                "splits": [],
                "note": f"Compound '{compound_clean}' not in sandhi data.",
            }

        return {
            "compound": compound_clean,
            "found": True,
            "splits": entries,
        }

    def meter_checker(self, split: str, episode: Dict[str, Any]) -> Dict[str, Any]:
        """Check if a sandhi split preserves the verse meter."""
        meter_data = episode.get("meter_data", {})
        split_clean = split.strip()

        entry = meter_data.get(split_clean)
        if entry is None:
            for key, val in meter_data.items():
                if key.lower() == split_clean.lower():
                    entry = val
                    break

        if entry is None:
            return {
                "split": split_clean,
                "found": False,
                "preserves_meter": None,
                "meter_name": None,
                "note": f"Split '{split_clean}' not in meter data.",
            }

        return {
            "split": split_clean,
            "found": True,
            "preserves_meter": entry.get("preserves_meter"),
            "meter_name": entry.get("meter_name"),
        }

    def commentary_fetch(
        self, term_or_verse: str, episode: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fetch a commentary fragment for a term or verse ID."""
        commentary = episode.get("commentary_data", {})
        key_clean = term_or_verse.strip()

        text = commentary.get(key_clean)
        if text is None:
            for key, val in commentary.items():
                if key.lower() == key_clean.lower():
                    text = val
                    break

        if text is None:
            return {
                "query": key_clean,
                "found": False,
                "commentary": None,
                "note": "No commentary available.",
            }

        return {
            "query": key_clean,
            "found": True,
            "commentary": text,
        }

    def witness_compare(
        self, verse_id: str, episode: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare variant readings from manuscript witnesses."""
        witness_data = episode.get("witness_data", {})
        vid_clean = verse_id.strip()

        entry = witness_data.get(vid_clean)
        if entry is None:
            for key, val in witness_data.items():
                if key.lower() == vid_clean.lower():
                    entry = val
                    break

        if entry is None:
            return {
                "verse_id": vid_clean,
                "found": False,
                "witnesses": [],
                "note": "No witness data available for this verse.",
            }

        return {
            "verse_id": vid_clean,
            "found": True,
            "witness_a": entry.get("witness_a", ""),
            "witness_b": entry.get("witness_b", ""),
            "agree": entry.get("witness_a") == entry.get("witness_b"),
        }

    def referent_tracker(
        self, pronoun: str, episode: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Track possible antecedents for a pronoun."""
        entity_map = episode.get("entity_map", {})
        pronoun_clean = pronoun.strip()

        candidates = entity_map.get(pronoun_clean)
        if candidates is None:
            for key, val in entity_map.items():
                if key.lower() == pronoun_clean.lower():
                    candidates = val
                    break

        if candidates is None:
            return {
                "pronoun": pronoun_clean,
                "found": False,
                "antecedents": [],
                "note": f"Pronoun '{pronoun_clean}' not tracked in entity map.",
            }

        return {
            "pronoun": pronoun_clean,
            "found": True,
            "antecedents": candidates,
        }

    @classmethod
    def catalog(cls) -> List[Dict[str, Any]]:
        """Return tool specifications for /tools/catalog endpoint."""
        return [
            {
                "name": "lexicon_lookup",
                "description": "Look up domain-specific meanings of a Sanskrit term from the episode glossary.",
                "input": {"type": "string", "description": "Sanskrit term in IAST"},
                "output": {"type": "object", "fields": ["term", "found", "in_passage", "meanings"]},
            },
            {
                "name": "sandhi_parser",
                "description": "Return all phonologically valid splits of a compound with splitting rules.",
                "input": {"type": "string", "description": "Sanskrit compound in IAST"},
                "output": {"type": "object", "fields": ["compound", "found", "splits"]},
            },
            {
                "name": "meter_checker",
                "description": "Check if a sandhi split preserves the verse meter (anustubh, tristubh, etc.).",
                "input": {"type": "string", "description": "Proposed split to check"},
                "output": {"type": "object", "fields": ["split", "found", "preserves_meter", "meter_name"]},
            },
            {
                "name": "commentary_fetch",
                "description": "Fetch a commentary fragment from a medieval Sanskrit commentary.",
                "input": {"type": "string", "description": "Term or verse ID"},
                "output": {"type": "object", "fields": ["query", "found", "commentary"]},
            },
            {
                "name": "witness_compare",
                "description": "Compare variant readings from two manuscript witnesses.",
                "input": {"type": "string", "description": "Verse ID"},
                "output": {"type": "object", "fields": ["verse_id", "found", "witness_a", "witness_b", "agree"]},
            },
            {
                "name": "referent_tracker",
                "description": "Track possible antecedents for a pronoun based on entities introduced so far.",
                "input": {"type": "string", "description": "Pronoun in IAST"},
                "output": {"type": "object", "fields": ["pronoun", "found", "antecedents"]},
            },
        ]
