from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import json


@dataclass
class FunctionCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelTurn:
    text: str | None = None
    tool_call: FunctionCall | None = None
    final: bool = False


class ToolRegistry:
    def __init__(
        self,
        notes_path: Path,
        google_search: Callable[[str], list[dict[str, str]]],
        url_context: Callable[[str], str],
    ) -> None:
        self.notes_path = notes_path
        self._google_search = google_search
        self._url_context = url_context
        self._tools: dict[str, Callable[..., Any]] = {
            "google_search": self.google_search,
            "url_context": self.url_context,
            "save_note": self.save_note,
            "summarize_notes": self.summarize_notes,
        }

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "google_search",
                "description": "Search for recent sources and return links.",
                "parameters": {"query": "string"},
            },
            {
                "name": "url_context",
                "description": "Read and extract context from a specific URL.",
                "parameters": {"url": "string"},
            },
            {
                "name": "save_note",
                "description": "Persist an important research note to disk.",
                "parameters": {"note": "string", "source": "string (optional)"},
            },
            {
                "name": "summarize_notes",
                "description": "Summarize collected notes for the team.",
                "parameters": {},
            },
        ]

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        return self._tools[name](**arguments)

    def google_search(self, query: str) -> list[dict[str, str]]:
        return self._google_search(query)

    def url_context(self, url: str) -> str:
        return self._url_context(url)

    def save_note(self, note: str, source: str | None = None) -> dict[str, str]:
        self.notes_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"note": note, "source": source or ""}
        with self.notes_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
        return {"status": "saved", "path": str(self.notes_path)}

    def summarize_notes(self) -> str:
        if not self.notes_path.exists():
            return "No notes have been saved yet."

        notes: list[str] = []
        with self.notes_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                source = obj.get("source", "")
                suffix = f" (source: {source})" if source else ""
                notes.append(f"- {obj.get('note', '')}{suffix}")

        if not notes:
            return "No notes have been saved yet."
        return "Research summary:\n" + "\n".join(notes)


class HeuristicResearchModel:
    """Simple prototype model that demonstrates tool-choice flow."""

    def next_turn(
        self,
        history: list[dict[str, Any]],
        available_tools: list[dict[str, Any]],
    ) -> ModelTurn:
        _ = available_tools
        tool_messages = [m for m in history if m["role"] == "tool"]

        if not tool_messages:
            user_prompt = next(m["content"] for m in history if m["role"] == "user")
            return ModelTurn(
                tool_call=FunctionCall(
                    name="google_search",
                    arguments={"query": user_prompt},
                )
            )

        seen_tools = [m["name"] for m in tool_messages]

        if "url_context" not in seen_tools:
            search_result = next(m for m in tool_messages if m["name"] == "google_search")
            results = search_result["content"]
            first_url = results[0]["url"]
            return ModelTurn(tool_call=FunctionCall(name="url_context", arguments={"url": first_url}))

        if "save_note" not in seen_tools:
            page_text = next(m["content"] for m in tool_messages if m["name"] == "url_context")
            note = page_text.split(".")[0].strip()
            return ModelTurn(
                tool_call=FunctionCall(
                    name="save_note",
                    arguments={"note": note, "source": "web context"},
                )
            )

        if "summarize_notes" not in seen_tools:
            return ModelTurn(tool_call=FunctionCall(name="summarize_notes", arguments={}))

        summary = next(m["content"] for m in tool_messages if m["name"] == "summarize_notes")
        return ModelTurn(text=summary, final=True)


class ConversationOrchestrator:
    def __init__(self, model: HeuristicResearchModel, tools: ToolRegistry, max_steps: int = 8) -> None:
        self.model = model
        self.tools = tools
        self.max_steps = max_steps

    def run(self, user_request: str) -> str:
        history: list[dict[str, Any]] = [{"role": "user", "content": user_request}]

        for _ in range(self.max_steps):
            turn = self.model.next_turn(history=history, available_tools=self.tools.schemas())

            if turn.tool_call:
                result = self.tools.call(turn.tool_call.name, turn.tool_call.arguments)
                history.append({"role": "tool", "name": turn.tool_call.name, "content": result})
                continue

            if turn.text:
                history.append({"role": "assistant", "content": turn.text})
                if turn.final:
                    return turn.text

        raise RuntimeError("Conversation ended before reaching a final response.")
