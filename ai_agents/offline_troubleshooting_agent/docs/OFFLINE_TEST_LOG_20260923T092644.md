# Offline Test Log — PASS

Run at: 2026-09-23T09:26:44.015261

```
======================================================================
OFFLINE TEST — network guard
======================================================================
Checking whether 1.1.1.1:53 is reachable...
OK — 1.1.1.1:53 unreachable within 2.0s. Proceeding as offline.

======================================================================
Local service health checks
======================================================================
Ollama / LLM:        {'reachable': True, 'model_available': True, 'local_models': ['qwen3:4b-instruct', 'glm-ocr:latest', 'llama3.2:latest', 'gemma4:e4b', 'moondream:latest', 'nomic-embed-text:latest', 'llama3.2:3b']}
Embedding model:     {'reachable': True, 'model_available': True, 'local_models': ['qwen3:4b-instruct', 'glm-ocr:latest', 'llama3.2:latest', 'gemma4:e4b', 'moondream:latest', 'nomic-embed-text:latest', 'llama3.2:3b']}
Actian VectorAI DB:  {'reachable': True, 'title': 'Actian VectorAI DB', 'version': 'Actian VectorAI DB 1.0.1 / VDE 1.0.1'}

======================================================================
Triggering bearing fault
======================================================================
Current readings: temperature_c=91.73767338498399 vibration_mm_s=7.911920233993246 pressure_bar=4.970595871281932 status='fault' error_code='BRG-02' timestamp=datetime.datetime(2026, 9, 23, 9, 24, 29, 279283)

======================================================================
Running investigation
======================================================================
  [1] tool_calls so far: 1  |  most recent: get_current_readings
  [2] tool_calls so far: 2  |  most recent: get_recent_history
  [3] tool_calls so far: 3  |  most recent: search_manual
  [4] tool_calls so far: 4  |  most recent: search_past_incidents
  [5] tool_calls so far: 5  |  most recent: search_manual
  [6] tool_calls so far: 6  |  most recent: search_past_incidents
  [7] tool_calls so far: 7  |  most recent: save_finding
  [8] tool_calls so far: 8  |  most recent: finish_investigation

final_result: {'likely_cause': 'Worn bearing in the main shaft assembly', 'confidence': 0.95, 'recommendation': 'Inspect the bearing assembly for visible wear, play, or discoloration. Check lubrication condition and top up or replace lubricant if low or contaminated. If excessive wear is confirmed, replace the bearing. After repair, restart and confirm vibration and temperature return to normal ranges before returning the machine to service.', 'supporting_evidence': ['Current vibration reading is 8.09 mm/s, which exceeds the normal 1.5–3.0 mm/s range.', 'The BRG-02 error code is active, matching symptoms of bearing vibration warning.', 'Past incident with identical symptoms (high vibration and rising temperature) confirmed a worn bearing as the cause.', 'Manual section on bearing troubleshooting recommends the same inspection and repair steps.'], 'retrieved_incident_ids': ['eaca15e8-b09a-51d3-852d-efc893955a49', 'c43aa763-6248-5175-96e3-5ddae4391342', 'bc455f13-34e4-4704-8854-0d9a86f40dc9'], 'manual_sources': ['BRG-02 — Bearing vibration warning']}
errors: []

Reached finish_investigation: Y
Investigation wall-clock time: 132.16s (2.2 min)

======================================================================
Checking recall of the offline-test reference incident
======================================================================
Baseline reference incident_id: bc455f13-34e4-4704-8854-0d9a86f40dc9
Baseline confirmed_cause: 'Worn bearing in the main shaft assembly'
search_incidents('bearing fault high vibration high temperature BRG-02') returned ids: ['eaca15e8-b09a-51d3-852d-efc893955a49', 'bc455f13-34e4-4704-8854-0d9a86f40dc9', '141954b3-97aa-4d33-91e1-7140b20cd76c', '4151d788-02fe-56ca-92f1-6744cd690d65', 'c43aa763-6248-5175-96e3-5ddae4391342']

PASS: reference incident bc455f13-34e4-4704-8854-0d9a86f40dc9 was recalled by exact id match.

======================================================================
FINAL SUMMARY
======================================================================
Reached finish_investigation:      Y
Correct prior incident recalled:   Y (id=bc455f13-34e4-4704-8854-0d9a86f40dc9)
state.errors:                      []
Total wall-clock time:              141.34s (2.4 min)

OVERALL: PASS
```
