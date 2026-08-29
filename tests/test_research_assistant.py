import json
import tempfile
import unittest
from pathlib import Path

from research_assistant import ConversationOrchestrator, HeuristicResearchModel, ToolRegistry


class ResearchAssistantTests(unittest.TestCase):
    def test_conversation_loop_uses_search_context_note_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            notes_path = Path(tmp) / "notes.jsonl"

            def fake_search(query: str):
                self.assertIn("recent", query.lower())
                return [{"title": "News", "url": "https://example.com/news"}]

            def fake_url_context(url: str):
                self.assertEqual(url, "https://example.com/news")
                return "AI funding increased by 20 percent this quarter. Extra details here."

            tools = ToolRegistry(notes_path=notes_path, google_search=fake_search, url_context=fake_url_context)
            orchestrator = ConversationOrchestrator(model=HeuristicResearchModel(), tools=tools)

            output = orchestrator.run("Find recent AI news and summarize for the team")

            self.assertIn("Research summary", output)
            self.assertIn("AI funding increased by 20 percent this quarter", output)
            self.assertTrue(notes_path.exists())

    def test_save_note_persists_json_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            notes_path = Path(tmp) / "notes.jsonl"
            tools = ToolRegistry(
                notes_path=notes_path,
                google_search=lambda q: [],
                url_context=lambda u: "",
            )

            tools.save_note("Important insight", source="https://example.com")

            lines = notes_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            saved = json.loads(lines[0])
            self.assertEqual(saved["note"], "Important insight")
            self.assertEqual(saved["source"], "https://example.com")


if __name__ == "__main__":
    unittest.main()
