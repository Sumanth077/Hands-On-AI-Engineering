# Offline Troubleshooting Agent with Actian VectorAI DB

## Full Implementation Reference

This document is the implementation reference for the coding agent. It expands the shorter project brief into a build-ready specification with enough detail to implement the project without guessing major product or architecture decisions.

The goal is to build a fully local AI troubleshooting assistant that can diagnose simulated equipment faults, remember previous incidents, and continue using that memory after the application restarts and the internet is unavailable.

The project should remain an AI engineering project, not an industrial-control project. The machine is simulated in Python. No real factory hardware, PLC integration, MQTT broker, OPC-UA, ROS stack, or industrial protocol is required for v1.

## 1. Core Idea

A simulated machine produces simple operational readings such as:

- temperature
- vibration
- pressure
- machine status
- error code
- timestamp

The user can trigger a fault from the local UI.

When a fault occurs, a local AI agent investigates by using a small set of tools:

- inspect current machine readings
- inspect recent sensor history
- search local equipment documentation
- search previous incidents stored in Actian VectorAI DB
- compare current evidence with relevant previous incidents
- produce a diagnosis and recommended action
- save the confirmed resolution as a new persistent memory

The important behavior is not simply local RAG over a manual. The agent should accumulate useful experience over time.

A resolved incident becomes context for future incidents. If the application is restarted, that memory must still exist. If internet connectivity is removed, the agent must still be able to retrieve and use it.

### Example

First incident:

```
Temperature: 92°C
Vibration: 8.1 mm/s
Pressure: 5.0 bar
Status: Warning
```

Agent investigates:
- recent temperature trend
- recent vibration trend
- local manual
- previous incidents

No strong prior incident exists.

Likely cause: Worn bearing

Technician confirms: Bearing replaced, readings returned to normal.

Incident is saved locally.

Later incident:

```
Temperature: 89°C
Vibration: 7.7 mm/s
Pressure: 5.1 bar
Status: Warning
```

Agent searches local memory.

Similar incident found: High temperature + high vibration, Cause: worn bearing, Fix: bearing replacement, Outcome: resolved

Recommendation: Inspect the bearing assembly first.

The second case should still work after restarting the app with internet access disabled.

## 2. What the Project Must Prove

### 2.1 Local troubleshooting

The agent can inspect machine data, local documentation, and local incident history without relying on cloud services.

### 2.2 Persistent memory

Resolved incidents are stored in Actian VectorAI DB and remain available across application restarts.

### 2.3 Offline operation

After all dependencies and local models have been installed, the running app can operate without internet access and still:

- launch the UI
- run the local LLM
- generate embeddings locally
- query Actian VectorAI DB
- read local manuals
- retrieve previous incidents
- perform a new troubleshooting session
- save a new incident

## 3. Non-Goals

Do not turn the project into a full industrial IoT platform.

The first version does not need:

- real PLC integration
- MQTT
- OPC-UA
- real factory equipment
- Raspberry Pi-specific code
- ROS
- computer vision
- audio input
- cloud LLM APIs
- cloud embedding APIs
- cloud-hosted vector databases
- predictive maintenance ML models
- real-time anomaly detection at production scale
- autonomous control of machinery
- multi-agent orchestration

The simulated machine exists only to create believable local events for the AI agent to investigate.

## 4. Recommended Tech Stack

- Python: core application, simulator, tools, agent loop
- Actian VectorAI DB: persistent local semantic memory and local vector retrieval
- Ollama: local LLM inference
- Local embedding model: offline embedding generation
- Gradio: local web interface
- Pandas: sensor history and tabular processing
- Pydantic: structured schemas and agent state

### Hardware assumption for v1

The reference build should target a normal developer laptop rather than dedicated edge hardware. It must remain usable without a discrete GPU. GPU acceleration may improve speed when available, but the application should not require CUDA or a dedicated accelerator to function. Keep model size, retrieval depth, prompt size, and concurrent work conservative so the demo remains practical on limited RAM.

### 4.1 Local LLM

Use `qwen3:4b-instruct` through Ollama as the default model for v1. The default Ollama quantized build is roughly 2.5 GB, which keeps the project realistic for a modest laptop while still providing much stronger instruction following and tool-use behavior than older 1B-3B models.

Default configuration:

```
Model: qwen3:4b-instruct
Runtime: Ollama
Quantization target: Q4_K_M / Ollama default quantized build
Approximate model file size: ~2.5 GB
```

Keep the model configurable through `OLLAMA_MODEL`, but build and test the reference implementation against `qwen3:4b-instruct`.

The project should be designed for constrained hardware rather than assuming a workstation-class GPU. Start with a modest context window and small retrieved payloads. Do not send entire manuals, full sensor histories, or every previous incident to the model. Retrieve only the few pieces of evidence needed for the current step.

Recommended low-resource defaults:

- keep the model context modest; start around 4K-8K tokens unless the laptop comfortably supports more
- retrieve about 3 relevant manual chunks at a time
- retrieve about 3 relevant past incidents at a time
- keep manual chunks compact, roughly 400-800 tokens
- summarize or truncate large tool outputs before returning them to the model
- keep only recent tool interactions plus compact structured investigation state
- avoid loading multiple LLMs simultaneously

If `qwen3:4b-instruct` is too slow on the target laptop, allow a documented lower-resource fallback such as a small Phi-4 Mini quantized build, but do not silently change the default model.

For the offline demo, the configured Ollama model must be a fully local model. Do not use Ollama cloud-backed model tags or any provider-backed fallback. Pull the model once during setup, then verify that inference still works after network access is removed.

### 4.2 Embeddings

Use a lightweight local embedding model. A practical default is `nomic-embed-text` through Ollama, which keeps both generation and embedding inference inside the same local runtime.

Requirements:

- no remote API calls during normal operation
- lightweight enough for a modest laptop
- suitable for semantic similarity over short incident summaries and manual chunks
- downloaded before the offline demo
- loadable and callable with the network disabled

Keep the embedding model configurable through `EMBEDDING_MODEL`, but use `nomic-embed-text` as the reference default unless implementation testing shows a compatibility problem.

### 4.3 Actian integration

Use the current official Actian VectorAI DB local deployment and Python SDK/client instructions at implementation time.

Do not invent package names, image tags, ports, or SDK methods if the current docs differ.

The integration must support:

- local database connection
- collection/index creation
- vector upsert
- metadata storage
- semantic similarity search
- persistent local storage across restarts

If Actian runs in Docker, use a mounted persistent volume so incidents are not lost when the container is restarted.

## 5. High-Level Architecture

```
┌──────────────────────────────────────────────┐
│                  Gradio UI                    │
│                                                │
│ Machine Status | Investigation | Local Memory │
└───────────────────────┬──────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────┐
│             Troubleshooting Agent             │
│                                                │
│ reasoning / tool choice / investigation state │
└───────────────┬────────────────────────────────┘
                 │
       ┌─────────┼──────────┬──────────────────┐
       │         │          │                   │
       ▼         ▼          ▼                   ▼
   Current    Sensor      Manual            Incident
   Readings   History     Search            Search
       │         │          │                   │
       └─────────┴──────────┴───────────────────┘
                         │
                         ▼
                Local data + Actian
                         │
                         ▼
                  Local Ollama model
```

No cloud service should be required during normal operation.

## 6. Suggested Repository Structure

```
offline_troubleshooting_agent/
│
├── README.md
├── requirements.txt
├── .env.example
├── app.py
│
├── agent/
│   ├── __init__.py
│   ├── harness.py
│   ├── prompts.py
│   ├── state.py
│   └── tools.py
│
├── simulator/
│   ├── __init__.py
│   ├── machine.py
│   ├── scenarios.py
│   └── history.py
│
├── memory/
│   ├── __init__.py
│   ├── vectorai_client.py
│   ├── embeddings.py
│   └── incident_store.py
│
├── documents/
│   ├── machine_manual.md
│   └── troubleshooting_guide.md
│
├── data/
│   ├── sensor_history.csv
│   └── seed_incidents.json
│
├── models/
│   └── schemas.py
│
├── ui/
│   ├── __init__.py
│   └── gradio_app.py
│
├── scripts/
│   ├── seed_memory.py
│   └── reset_demo.py
│
└── tests/
    ├── test_simulator.py
    ├── test_memory.py
    ├── test_tools.py
    ├── test_agent.py
    └── test_offline_assumptions.py
```

The exact folder structure may be simplified, but keep the simulator, memory layer, agent logic, and UI separated.

## 7. Machine Simulator

The simulator should be intentionally simple and deterministic enough for demos and tests.

### 7.1 Machine state

Recommended fields:

```
temperature_c: float
vibration_mm_s: float
pressure_bar: float
status: str
error_code: str | None
timestamp: datetime
```

Suggested status values:

```
running
warning
fault
maintenance
stopped
```

### 7.2 Normal operating range

Example baseline:

```
Temperature: 65–75°C
Vibration: 1.5–3.0 mm/s
Pressure: 4.5–5.5 bar
Status: running
```

These values are illustrative and are not meant to represent one specific real industrial machine.

### 7.3 Fault scenarios

Implement at least three fault types so the agent cannot simply memorize one response.

#### Scenario A: Worn bearing

Pattern:

- temperature rises
- vibration rises strongly
- pressure stays normal

Example state:

```
Temperature: 92°C
Vibration: 8.0 mm/s
Pressure: 5.0 bar
Error code: BRG-02
```

Expected likely diagnosis: bearing wear / bearing issue

Expected maintenance action: inspect bearing assembly, replace worn bearing if confirmed

#### Scenario B: Cooling problem

Pattern:

- temperature rises
- vibration remains near normal
- pressure stays near normal

Example state:

```
Temperature: 96°C
Vibration: 2.3 mm/s
Pressure: 5.1 bar
Error code: TMP-04
```

Expected likely diagnosis: cooling airflow / cooling system problem

Expected action: inspect cooling fan, vents, coolant path, or airflow

#### Scenario C: Pressure loss

Pattern:

- pressure drops
- temperature may be mildly elevated
- vibration remains near normal

Example state:

```
Temperature: 77°C
Vibration: 2.4 mm/s
Pressure: 2.8 bar
Error code: PRS-03
```

Expected likely diagnosis: pressure leak / pressure system issue

Expected action: inspect seals, hoses, fittings, or pressure path

### 7.4 Simulator API

Expose functions similar to:

```
get_current_readings()
get_recent_history(limit=20)
trigger_fault(fault_type)
reset_machine()
advance_simulation(steps=1)
```

The UI should allow the user to:

- begin with normal operation
- trigger one of the fault scenarios
- reset the machine
- optionally generate a short history leading into the fault

## 8. Local Equipment Documentation

Create a small fictional machine manual specifically for the demo.

Do not use proprietary real-world manuals.

The manual should include:

- normal temperature range
- normal vibration range
- normal pressure range
- error code descriptions
- bearing troubleshooting guidance
- cooling troubleshooting guidance
- pressure troubleshooting guidance
- basic safety disclaimer

Example manual entry:

```
BRG-02 — Bearing vibration warning

Symptoms:
- elevated vibration
- possible temperature increase

Recommended inspection:
1. stop the machine according to normal shutdown procedure
2. inspect bearing assembly
3. inspect lubrication condition
4. replace bearing if excessive wear is confirmed
```

Split the manual into retrievable chunks during ingestion.

Manual chunks and incident memories should remain logically separate even if they are stored in the same database instance.

## 9. Incident Memory Model

A resolved incident should be stored as structured metadata plus an embedding-friendly text representation.

### 9.1 Suggested Pydantic schema

```python
class Incident(BaseModel):
    incident_id: str
    created_at: datetime
    machine_id: str

    symptoms: list[str]

    temperature_c: float | None
    vibration_mm_s: float | None
    pressure_bar: float | None
    error_code: str | None

    diagnosis: str
    confirmed_cause: str
    fix_applied: str
    outcome: str

    notes: str | None = None
```

### 9.2 Embedding text

Build a semantic representation similar to:

```
Machine: Machine A
Symptoms: high temperature, high vibration
Readings: temperature 92 C, vibration 8.1 mm/s, pressure 5.0 bar
Error code: BRG-02
Confirmed cause: worn bearing
Fix applied: bearing replacement
Outcome: resolved
```

Generate the embedding locally and store the vector together with the structured metadata in Actian.

## 10. VectorAI DB Memory Design

Prefer two logical collections/indexes if that maps cleanly to the current Actian API.

### 10.1 manual_chunks

Store:

- chunk text
- section title
- source filename
- optional component metadata
- embedding vector

### 10.2 incidents

Store:

- semantic incident text
- incident ID
- timestamp
- diagnosis/fault type
- readings metadata
- confirmed resolution metadata
- embedding vector

If the current Actian setup prefers one collection plus metadata filtering, that is acceptable. Keep the logical distinction in application code.

### 10.3 Similarity query

For a new fault, build a query such as:

```
Machine fault with high temperature 91 C, high vibration 7.8 mm/s,
normal pressure, error code BRG-02.
```

Retrieve the most relevant incidents.

Return to the agent:

- incident ID
- relevance/similarity score if available
- summary
- confirmed cause
- previous fix
- timestamp

The top vector match must not automatically become the diagnosis. Retrieved memory is evidence, not truth.

## 11. Agent Responsibilities

The model should decide:

- what evidence it needs
- which tool to call next
- whether a previous incident is relevant
- whether the manual supports or contradicts a possible explanation
- whether the current evidence conflicts with retrieved memory
- what diagnosis is most likely
- what action to recommend
- when enough evidence exists to finish

Do not put the diagnosis inside the tools themselves.

Avoid tools like:

```
diagnose_bearing_fault()
find_root_cause()
choose_fix()
```

Those would hide the interesting reasoning inside application logic.

## 12. Agent Tools

Recommended tool surface:

**get_current_readings()** — Returns current simulated machine readings.

**get_recent_history(limit=20)** — Returns recent readings in chronological order.

**search_manual(query, top_k=3)** — Embeds the query locally and searches local manual chunks in Actian.

**search_past_incidents(query, top_k=3)** — Searches incident memory in Actian.

**save_finding(finding, evidence)** — Adds an important observation to the current investigation state.

**save_incident(...)** — Called only after the user/technician confirms the real cause and fix.

Save both:

- structured incident metadata
- semantic text + embedding

**finish_investigation(...)** — Returns structured final output.

Suggested fields:

```
likely_cause: str
confidence: float
recommendation: str
supporting_evidence: list[str]
retrieved_incident_ids: list[str]
manual_sources: list[str]
```

## 13. Investigation State

Suggested state model:

```python
class InvestigationState(BaseModel):
    investigation_id: str
    objective: str

    current_readings: dict
    findings: list[str]

    retrieved_incidents: list[dict]
    retrieved_manual_chunks: list[dict]

    tool_calls: list[dict]
    errors: list[str]

    final_result: dict | None = None
```

Keep the state small and explicit.

The first version does not need a large planning framework.

## 14. Agent Loop

A minimal loop is enough.

```
Receive fault / objective
        ↓
Send current state + tools to local LLM
        ↓
Model selects tool(s)
        ↓
Execute locally
        ↓
Return tool result
        ↓
Update investigation state
        ↓
Repeat
        ↓
finish_investigation()
```

Include:

- maximum model rounds
- maximum tool calls
- basic duplicate-call detection
- graceful tool error handling

Do not overengineer reflection, planning graphs, or multi-agent behavior in v1.

## 15. Prompt Requirements

The system prompt should instruct the local model to:

- act as a local equipment troubleshooting assistant
- inspect evidence before diagnosing
- never invent sensor readings
- treat previous incidents as context, not guaranteed truth
- use the manual where useful
- distinguish current evidence from previous experience
- avoid overconfidence when evidence is weak
- recommend inspections rather than unsafe control actions
- never instruct the user to bypass safety mechanisms
- never directly control machinery
- stop when enough evidence exists for a reasonable recommendation

This is a simulation/demo assistant, not a certified industrial safety system.

## 16. Gradio Interface

Use Gradio locally.

Do not require `share=True` or hosted Hugging Face features.

The interface should have four obvious areas.

### 16.1 Machine Status

Display:

```
Machine A
Temperature
Vibration
Pressure
Status
Error code
```

Controls:

- Trigger Bearing Fault
- Trigger Cooling Fault
- Trigger Pressure Fault
- Reset Machine

A small chart of recent readings is optional but useful.

### 16.2 Investigation

Show only high-level agent activity and tool usage, not hidden chain-of-thought.

Example:

```
✓ Checked current readings
✓ Reviewed recent history
✓ Searched equipment manual
✓ Searched previous incidents
● Comparing evidence...
```

Also show concise findings as they are saved.

### 16.3 Retrieved Memory

Show relevant previous incidents.

Example:

```
Similar incident
Date: 2026-09-10
Symptoms: high temperature + high vibration
Cause: worn bearing
Fix: bearing replacement
Similarity: 0.87
```

If none are useful:

```
No similar previous incident found.
```

### 16.4 Recommendation

Display:

- likely cause
- confidence
- recommendation
- supporting evidence
- manual references
- past incidents used

## 17. Incident Confirmation Flow

Do not let the agent automatically write its own diagnosis into permanent memory as if it were ground truth.

After the investigation, let the user confirm or correct the resolution.

Example:

```
Agent diagnosis:
Worn bearing

Confirmed cause:
[ Worn bearing ]

Fix applied:
[ Bearing replaced ]

Outcome:
[ Resolved ]

[ Save resolved incident ]
```

The confirmed resolution is what should be written to Actian as durable memory.

This prevents a bad AI guess from contaminating future retrieval.

## 18. Seed Data

Support two useful demo states.

### 18.1 Fresh state

No previous incidents.

Use this for the first troubleshooting run.

### 18.2 Seeded state

Contains a few resolved incidents, for example:

- bearing failure
- cooling fan blockage
- pressure leak
- harmless vibration caused by a loose external panel

Seed memories should be similar enough that semantic retrieval has to rank them meaningfully.

Provide a reset script to clear or reseed the demo.

## 19. Primary Demo Scenario

The main demo should use the bearing fault.

### Part 1: First incident

1. Launch the local Gradio app.
2. Start with normal readings.
3. Trigger the bearing fault.
4. Start an investigation.
5. Agent checks readings and history.
6. Agent searches the manual.
7. Agent searches incident memory.
8. No strong prior memory exists.
9. Agent recommends inspecting the bearing.
10. User confirms the cause was a worn bearing.
11. User confirms bearing replacement fixed it.
12. Save the incident to Actian.

### Part 2: Persistence test

1. Stop the Gradio application.
2. Keep Actian persistent storage intact.
3. Restart the application.
4. Confirm the saved incident still appears in memory/search.

### Part 3: Offline test

1. Before disconnecting, pull `qwen3:4b-instruct` and the configured embedding model locally and run one successful inference/embedding request to confirm both are cached.
2. Disable Wi-Fi or disconnect the network.
3. Restart or continue the local application.
4. Trigger a similar bearing fault.
5. Start a new investigation.
6. Agent searches Actian locally.
7. Previous bearing incident is retrieved.
8. Agent uses it as supporting context.
9. Recommendation still works without internet.

This is the main proof point of the project.

## 20. Offline Requirements

The coding agent must actively avoid accidental cloud dependencies.

Before calling the project offline-capable, verify:

- Gradio loads on localhost without internet
- `qwen3:4b-instruct` responds through local Ollama with internet disabled
- no Ollama cloud-backed model tag or provider fallback is configured
- the embedding model loads and generates embeddings locally with internet disabled
- Actian VectorAI DB is reachable locally
- manual retrieval works locally
- incident retrieval works locally
- saved incident survives restart
- no cloud API key is required for the core workflow

If a model or Python library needs to download assets on first use, document that as a one-time setup step before going offline.

Offline means the running application works after setup, not that installation can happen without internet.

## 21. Offline Status Display

The UI may show:

```
Deployment mode: LOCAL
Internet status: OFFLINE
Memory: Actian VectorAI DB
LLM: Ollama
Embeddings: Local
```

Do not make the app depend on a cloud service just to determine whether it is offline.

Use a short connectivity check or a manual demo toggle if automatic network detection adds unnecessary fragility.

The actual proof comes from running the demo disconnected, not from the badge itself.

## 22. Persistence Test

At minimum, manually or automatically validate:

```
save incident
↓
stop application
↓
restart application
↓
query incident memory
↓
incident still exists
```

If Actian runs in Docker, data must live in a persistent volume rather than the container filesystem.

## 23. Memory Retrieval Tests

Test semantic memory with several cases.

### Exact repeat

Query: `high temperature + high vibration + BRG-02`

Expected: bearing incident ranks highly.

### Similar wording

Query: `machine running hot and shaking much more than normal`

Expected: bearing incident still retrieves.

### Conflicting case

Query: `high temperature + normal vibration`

Expected: cooling incident should outrank bearing memory if both exist.

### Unrelated fault

Query: `low pressure + normal vibration`

Expected: pressure incident should rank ahead of bearing memory.

The agent should not blindly accept the top vector result when current evidence conflicts with it.

## 24. Error Handling

Handle these cases cleanly:

- Actian unavailable
- Ollama unavailable
- embedding model not downloaded
- no manual matches
- no incident matches
- malformed model output
- tool exception
- duplicate tool call
- max investigation rounds reached

The UI should surface concise operational errors without crashing the whole application.

## 25. Safety Boundaries

This project is a simulation/demo.

The agent must not:

- directly control machinery
- disable alarms
- bypass interlocks
- recommend unsafe physical actions
- claim to replace trained maintenance staff

Recommended wording:

```
Inspect the bearing assembly.
Compare the current pattern with the previous bearing incident.
Follow the equipment's normal shutdown and maintenance procedure.
```

Avoid unsafe wording such as:

```
Keep the machine running and remove the bearing guard.
```

## 26. Automated Tests

Implement tests for the following.

### Simulator

- normal readings are within expected ranges
- each fault scenario creates the intended pattern
- reset returns the machine to normal

### Incident model

- required fields are enforced
- incidents serialize and deserialize correctly

### Memory

- incident can be inserted
- incident can be retrieved
- similar wording retrieves relevant memory
- metadata survives round-trip
- persistence survives restart where practical

### Manual retrieval

- bearing query retrieves bearing guidance
- cooling query retrieves cooling guidance
- pressure query retrieves pressure guidance

### Tools

- current readings tool
- recent history tool
- manual search tool
- incident search tool
- incident save tool

### Agent behavior

Use mocked LLM output where useful so CI remains deterministic.

### Offline assumptions

Verify configured endpoints are local and the core app requires no cloud API key.

## 27. Environment Configuration

Suggested `.env.example`:

```
# Actian VectorAI DB
ACTIAN_VECTORAI_HOST=
ACTIAN_VECTORAI_PORT=
ACTIAN_VECTORAI_INCIDENT_COLLECTION=
ACTIAN_VECTORAI_MANUAL_COLLECTION=

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:4b-instruct

# Local embeddings
EMBEDDING_MODEL=nomic-embed-text

# Agent limits
MAX_AGENT_ROUNDS=8
MAX_TOOL_CALLS=20

# Gradio
GRADIO_SERVER_NAME=127.0.0.1
GRADIO_SERVER_PORT=7860
```

Use the actual current Actian environment/configuration fields required by the official client.

Do not commit secrets.

## 28. README Requirements

The README should explain:

- what the project does
- why local persistent memory matters
- architecture
- prerequisites
- how to start Actian VectorAI DB locally
- how to install/download the Ollama model
- how to download/cache the embedding model
- how to seed demo data
- how to launch Gradio
- how to run the first incident demo
- how to perform the persistence test
- how to perform the offline test
- how to reset the demo
- limitations
- safety note

Keep the README focused on reproducibility.

## 29. Acceptance Criteria

The project is ready when all of the following are true.

### Functional

- Gradio UI launches locally
- machine simulator shows normal readings
- at least 3 fault scenarios can be triggered
- agent can inspect current readings
- agent can inspect recent history
- agent can search local manual content
- agent can retrieve previous incidents
- agent can produce a structured recommendation
- user can confirm/correct the real cause and fix
- confirmed incident is stored in Actian

### Persistence

- saved incident survives application restart
- saved incident survives Actian container restart when using a persistent volume

### Offline

- application starts with internet disabled after setup
- Ollama works offline
- embeddings work offline
- VectorAI search works offline
- manual retrieval works offline
- incident retrieval works offline
- new incident can be saved offline

### Demo quality

- first bearing incident can be investigated and saved
- second similar bearing incident retrieves the first
- UI visibly shows retrieved memory
- current evidence is separated from past-incident evidence
- recommendation does not simply copy memory when current readings conflict

## 30. Nice-to-Have Features

Only add these after the core demo works reliably.

- line chart of recent sensor readings
- similarity scores in memory panel
- incident timeline
- export incident report to Markdown
- multiple simulated machines
- automatic fault stream
- saved investigation history
- resolved-incident dashboard
- manual offline demo badge
- selectable Ollama model
- selectable embedding model

Do not let nice-to-have features delay persistence and offline validation.

## 31. Implementation Priorities

Build in this order:

1. machine simulator
2. local Gradio UI
3. local Ollama connection
4. local embedding model
5. Actian VectorAI DB connection
6. manual ingestion and retrieval
7. incident storage and retrieval
8. agent tools
9. agent loop
10. confirmation + save flow
11. persistence test
12. offline test
13. automated tests
14. README polish

Do not start with styling, charts, or complex agent planning.

## 32. Final User Experience

The finished project should be understandable without explaining the architecture first.

A user opens the local Gradio app and sees a simulated machine operating normally.

They trigger a fault.

The agent investigates using local readings, local documentation, and Actian memory.

The user confirms the real cause and fix.

The incident is saved.

The app is closed.

The network is disconnected.

The app is opened again.

A similar fault is triggered.

The agent retrieves the earlier incident and uses it as context for the new diagnosis.

The project succeeds when the demo makes this sentence visibly true:

**The AI can lose the internet without losing its memory.**
