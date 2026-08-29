# Google-AI-agent-architecture-design

Prototype implementation of an AI research assistant that combines:
- Gemini built-in style tools (`google_search`, `url_context`)
- Custom tools (`save_note`, `summarize_notes`)
- A conversation loop that executes model-selected function calls until a final summary is produced

## Files
- `research_assistant.py` — core architecture (tool registry, custom functions, model-driven function-calling loop)
- `tests/test_research_assistant.py` — focused tests for multi-step orchestration and note persistence

## Run tests
```bash
python -m unittest discover -s tests -v
```
