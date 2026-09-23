# Build Log

Progress notes for the Offline Troubleshooting Agent build, tracked against the
14-step implementation order in `PROJECT_REFERENCE.md` section 31. This file is
a working log, not user-facing documentation — see the README (step 14) for
that once it exists.

## Status against section 31's 14 steps

Done:

1. Machine simulator — `simulator/machine.py`, `simulator/scenarios.py`, `simulator/history.py`
2. Local Gradio UI (machine status only) — `ui/gradio_app.py`, `app.py`
3. Local Ollama connection — `agent/llm_client.py`
4. Local embedding model — `memory/embeddings.py`
5. Actian VectorAI DB connection — `memory/vectorai_client.py`, `docker-compose.yml`
6. Manual ingestion and retrieval — `memory/manual.py`, `documents/*.md`
7. Incident storage and retrieval — `memory/incident_store.py`, `models/schemas.py` (`Incident`), `data/seed_incidents.json`
8. Agent tools — `agent/tools.py`, `agent/state.py`
9. Agent loop — `agent/harness.py`, `agent/prompts.py`, plus Gradio wiring for Investigation/Retrieved Memory/Recommendation panels

Remaining:

10. Confirmation + save flow — human-in-the-loop UI that calls `memory.incident_store.save_incident()` after a technician confirms the real cause and fix (per section 17). Not built yet; `save_incident` is not reachable from the agent loop at all right now (see decision below).
11. Persistence test — save an incident, restart the app, confirm it's still retrievable (section 22).
12. Offline test — full network-disabled run-through per section 19 Part 3 / section 20.
13. Automated tests — the test suite already covers most of sections 26's checklist incrementally as each piece was built (simulator, memory, manual retrieval, tools, agent behavior with mocked LLM output), but hasn't had a final pass against the full section 26 list, and `tests/test_offline_assumptions.py` doesn't exist yet.
14. README polish — current `README.md` is still the step-1 placeholder.

## Key deviations and fixes made along the way

- **Docker service renamed to avoid a port collision.** The original `docker-compose.yml` used service/container name `vectorai` on host ports 6573-6575, which collided with an unrelated project's already-running container of the same name on the same ports. Renamed to `offline-troubleshooting-vectorai`, remapped to host ports 6673-6675 (container-internal ports unchanged), and updated `.env`'s `ACTIAN_VECTORAI_PORT` to `6674` accordingly. This project's own `local_data/` volume was verified to be freshly created and isolated from the other project's data.
- **Shared `_normalize_tag` helper for untagged model names.** Ollama's `client.list()` returns fully-tagged names (e.g. `nomic-embed-text:latest`), but `.env` often configures a model without an explicit tag (e.g. `EMBEDDING_MODEL=nomic-embed-text`). A naive string-equality check between the two reports `model_available: False` even when the model is actually pulled. Fixed with a shared `_normalize_tag(name)` helper (`name` → `name:latest` when no tag is present) in `agent/llm_client.py`, reused by `memory/embeddings.py`'s `embedding_health_check()` and `agent/llm_client.py`'s own `health_check()`. Covered by a regression test in `tests/test_llm_client.py`.
- **Protobuf version conflict resolved with a project-local `.venv` (discovered in step 5).** Installing `actian-vectorai-client` pulled in a `protobuf` version that conflicted with other tooling already installed outside the project. Rather than fight the conflict, the project was pinned to its own isolated `.venv/`, created before `pip install -r requirements.txt` and activated for every subsequent step. This keeps the project's dependency set fully isolated from whatever else is installed on the machine, and is now documented as its own setup step in the README.
- **UUID-only `PointStruct.id` constraint (discovered in step 5).** The real `actian-vectorai-client` SDK only accepts a non-negative int or a valid UUID string as a point id — an arbitrary string (e.g. `"inc-1"`) raises a `ValidationError`. This directly shapes `Incident.incident_id`: it must always be generated as `str(uuid.uuid4())` for new incidents, or a deterministic `str(uuid.uuid5(uuid.NAMESPACE_URL, ...))` for reproducible seed data (used in `data/seed_incidents.json` and manual-chunk ids in `memory/manual.py`).
- **JSON-based duplicate-call signature (step 9).** The agent loop's duplicate-tool-call detection originally built a signature as `(name, tuple(sorted(args.items())))`, which crashes with `TypeError: unhashable type: 'list'` the moment any tool argument is a list — which `finish_investigation` always has (`supporting_evidence`, `retrieved_incident_ids`, `manual_sources` are all lists). Fixed by using a stable JSON string (`json.dumps(args, sort_keys=True, default=str)`) as the signature instead, in `agent/harness.py`.

## Architectural decision: `save_incident` is never model-callable

Per section 17, nothing may be written to permanent memory until a human
confirms it. `memory.incident_store.save_incident()` is deliberately **not**
included in `AgentTools.tool_list()` and therefore never reaches
`ollama.chat(tools=...)` — the model has no way to invoke it, directly or
indirectly. This is locked in with an explicit test
(`tests/test_tools.py::test_tool_list_does_not_include_save_incident`) so it
can't regress silently in a later step. `save_incident` will only be called
by step 10's human-confirmation UI flow, after a technician reviews and
confirms (or corrects) the agent's `finish_investigation` output.

## Current blocker

`qwen3:4b-instruct` (the project's configured default `OLLAMA_MODEL`) has not
been pulled yet on this machine — the pull was started but is slow over the
current connection. As a result:

- `scripts/check_agent.py` correctly exits early with a clear "model not
  pulled" message when run against the real configured default, rather than
  failing partway through.
- Step 9's agent loop (`agent/harness.py`) and its Gradio wiring were
  validated for real end-to-end behavior using two substitute local models
  instead: `llama3.2:3b` (weaker — didn't reliably call
  `finish_investigation`, correctly triggered the fallback path) and
  `gemma4:e4b` (stronger — completed a full clean investigation through to
  `finish_investigation`, all three UI panels populated correctly).
- This gives good confidence the harness logic itself is correct (also
  covered by fully-mocked unit tests in `tests/test_agent.py`), but step 9
  should not be treated as **fully** closed until one real run against the
  actual configured default (`qwen3:4b-instruct`) has been done via
  `python scripts/check_agent.py`.

## Docker container note

Two Actian VectorAI containers currently exist on this machine, for two
unrelated projects:

- **This project**: `offline-troubleshooting-vectorai`, host ports
  `6673-6675`, backed by this project's own `local_data/` directory.
- **A different project** (`actian project/clinical-search`): container
  `vectorai`, host ports `6573-6575`, backed by that project's own
  `local_data/` directory.

Do not stop, restart, rename, or otherwise touch the `vectorai` container —
it belongs to the other project and is unrelated to this one.
