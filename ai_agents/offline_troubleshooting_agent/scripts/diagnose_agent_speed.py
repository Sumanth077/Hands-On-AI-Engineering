"""Diagnostic-only script: why is a real investigation against
qwen3:4b-instruct so slow on this hardware?

This does NOT fix anything. It measures five things in order and prints a
report as it goes, so partial results are visible even if a later step
hangs. See the numbered sections below.
"""

import json
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.llm_client import (  # noqa: E402
    _configured_base_url,
    _configured_model,
    assert_local_only,
    get_client,
)
from agent.prompts import SYSTEM_PROMPT, build_kickoff_message  # noqa: E402
from agent.state import InvestigationState  # noqa: E402
from agent.tools import AgentTools  # noqa: E402
from simulator.machine import Machine  # noqa: E402
from simulator.scenarios import FAULT_TARGETS, trigger_fault  # noqa: E402


def _header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78, flush=True)


def _thinking_info(message) -> tuple[bool, int]:
    thinking = getattr(message, "thinking", None)
    length = len(thinking) if thinking else 0
    return bool(thinking), length


def step1_baseline() -> float:
    _header("STEP 1: Baseline bare-model latency (no tools attached)")
    client = get_client()
    model = _configured_model()
    print(f"model={model}")

    messages = [{"role": "user", "content": "Say OK."}]
    start = time.monotonic()
    response = client.chat(model=model, messages=messages, think=False)
    elapsed = time.monotonic() - start

    has_thinking, thinking_len = _thinking_info(response.message)

    print(f"\nElapsed: {elapsed:.2f}s")
    print(f"Response content: {response.message.content!r}")
    print(f"message.thinking populated despite think=False: {has_thinking} (length={thinking_len} chars)")
    if has_thinking:
        print(f"thinking excerpt: {response.message.thinking[:300]!r}")
    return elapsed


def step2_with_tools() -> float:
    _header("STEP 2: Bare model + full tool schema, still one round")
    client = get_client()
    model = _configured_model()

    machine = Machine(seed=99)
    state = InvestigationState(
        investigation_id="diag-step2",
        objective="diagnostic",
        current_readings=machine.get_current_readings().model_dump(mode="json"),
    )
    tools = AgentTools(machine=machine, history=machine.history, state=state)
    tool_list = tools.tool_list()
    print(f"attaching {len(tool_list)} tool schemas: {[fn.__name__ for fn in tool_list]}")

    messages = [{"role": "user", "content": "Say OK."}]
    start = time.monotonic()
    response = client.chat(model=model, messages=messages, tools=tool_list, think=False)
    elapsed = time.monotonic() - start

    has_thinking, thinking_len = _thinking_info(response.message)

    print(f"\nElapsed: {elapsed:.2f}s")
    print(f"Response content: {response.message.content!r}")
    print(f"tool_calls made (unprompted, unusual if any): {response.message.tool_calls}")
    print(f"message.thinking populated: {has_thinking} (length={thinking_len} chars)")
    return elapsed


def step3_model_residency() -> None:
    _header("STEP 3: Is the model staying loaded between calls?")
    client = get_client()
    model = _configured_model()

    keep_alive_env = os.getenv("OLLAMA_KEEP_ALIVE")
    print(f"OLLAMA_KEEP_ALIVE env var: {keep_alive_env!r} (Ollama's own default is 5m if unset)")

    ps1 = client.ps()
    print("\nollama.ps() immediately after step 1/2 calls:")
    if not ps1.models:
        print("  (no models currently loaded)")
    for m in ps1.models:
        print(f"  - {m.model}: size_vram={getattr(m, 'size_vram', 'n/a')}, expires_at={getattr(m, 'expires_at', 'n/a')}")

    print("\nWaiting 10 seconds...")
    time.sleep(10)

    ps2 = client.ps()
    print("ollama.ps() after ~10s pause:")
    if not ps2.models:
        print("  (no models currently loaded)")
    for m in ps2.models:
        print(f"  - {m.model}: size_vram={getattr(m, 'size_vram', 'n/a')}, expires_at={getattr(m, 'expires_at', 'n/a')}")

    still_loaded = any(model in m.model or m.model in model for m in ps2.models)
    print(f"\n{model!r} still shown as loaded after 10s pause: {still_loaded}")

    # Cross-check against Ollama's own server log for actual reload events —
    # more reliable than inferring eviction from two ps() snapshots 10s apart,
    # since real investigation rounds are spaced much further apart than 10s.
    log_path = os.path.join(
        os.environ.get("LOCALAPPDATA", ""), "Ollama", "server.log"
    )
    print(f"\nChecking Ollama server log for model (re)load events: {log_path}")
    if os.path.isfile(log_path):
        try:
            with open(log_path, encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
            load_lines = [
                line.strip() for line in lines if "load_tensors: offloaded" in line
            ]
            print(f"Found {len(load_lines)} 'load_tensors: offloaded' events in the log so far this session.")
            print("Last 10 (each one is a full model (re)load into memory/VRAM):")
            for line in load_lines[-10:]:
                print(f"  {line}")
            if len(load_lines) >= 2:
                print(
                    "\nIf this count keeps climbing by more than 1 per investigation "
                    "round, the model is being evicted and reloaded repeatedly — "
                    "each reload re-reads the model file from disk and re-uploads "
                    "layers to GPU, which is real wall-clock cost on top of inference."
                )
        except OSError as exc:
            print(f"Could not read log file: {exc}")
    else:
        print("Log file not found at that path — skipping log cross-check.")


def _message_char_count(messages: list) -> int:
    total = 0
    for m in messages:
        if isinstance(m, dict):
            total += len(str(m.get("content", "")))
        else:
            total += len(str(getattr(m, "content", "") or ""))
    return total


def step4_instrumented_investigation() -> None:
    _header("STEP 4: Real investigation, instrumented (bearing fault)")
    print(
        "Note: this is a throwaway, instrumented re-implementation of "
        "agent/harness.py's loop for measurement only — it omits the "
        "no-tool-call nudge retry (unrelated to timing) and is NOT a "
        "replacement for the real harness. agent/harness.py itself is "
        "untouched.\n"
    )

    machine = Machine(seed=7)
    trigger_fault(machine, "bearing")
    machine.advance_simulation(steps=FAULT_TARGETS["bearing"].total_steps + 2)

    objective = "Investigate bearing fault on Machine A"
    current_readings = machine.get_current_readings().model_dump(mode="json")
    state = InvestigationState(
        investigation_id="diag-step4",
        objective=objective,
        current_readings=current_readings,
    )
    tools = AgentTools(machine=machine, history=machine.history, state=state)
    dispatch = {fn.__name__: fn for fn in tools.tool_list()}

    base_url = _configured_base_url()
    model = _configured_model()
    assert_local_only(base_url, model)
    client = get_client()

    messages: list = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_kickoff_message(objective, current_readings)},
    ]

    max_rounds = int(os.getenv("MAX_AGENT_ROUNDS", 8))
    max_tool_calls = int(os.getenv("MAX_TOOL_CALLS", 20))
    executed_signatures: set[str] = set()
    total_tool_calls = 0
    round_chat_times: list[float] = []

    investigation_start = time.monotonic()

    for round_num in range(1, max_rounds + 1):
        msg_chars = _message_char_count(messages)
        print(f"\n--- Round {round_num} ---")
        print(f"messages so far: {len(messages)} entries, ~{msg_chars} chars total (rough token proxy)")

        chat_start = time.monotonic()
        response = client.chat(model=model, messages=messages, tools=list(dispatch.values()), think=False)
        chat_elapsed = time.monotonic() - chat_start
        round_chat_times.append(chat_elapsed)
        print(f"ollama.chat() wall-clock time: {chat_elapsed:.2f}s")

        has_thinking, thinking_len = _thinking_info(response.message)
        print(f"message.thinking populated: {has_thinking} (length={thinking_len} chars)")

        messages.append(response.message)
        tool_calls = response.message.tool_calls

        if not tool_calls:
            content = response.message.content or ""
            print(f"No tool calls this round. content: {content[:300]!r}")
            print("(the real harness would attempt one nudge retry here — skipped in this diagnostic)")
            break

        for call in tool_calls:
            if total_tool_calls >= max_tool_calls:
                print("max_tool_calls reached mid-round — stopping.")
                break

            name = call.function.name
            args = dict(call.function.arguments or {})
            signature = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
            print(f"  tool_call: {name}({args})")

            if signature in executed_signatures:
                print("    -> duplicate signature, skipped (not re-executed)")
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": name,
                        "content": "Skipped: identical call already made this investigation.",
                    }
                )
                total_tool_calls += 1
                continue

            executed_signatures.add(signature)
            fn = dispatch.get(name)
            tool_start = time.monotonic()
            try:
                result = fn(**args) if fn else f"Unknown tool '{name}'"
                messages.append({"role": "tool", "tool_name": name, "content": str(result)})
            except Exception as exc:  # noqa: BLE001
                messages.append({"role": "tool", "tool_name": name, "content": f"Tool error: {exc}"})
            tool_elapsed = time.monotonic() - tool_start
            print(f"    tool execution time (Python function, not the model): {tool_elapsed:.3f}s")

            total_tool_calls += 1

        if state.final_result is not None:
            print("\nfinish_investigation reached — stopping instrumented loop.")
            break

        if total_tool_calls >= max_tool_calls:
            print("\nmax_tool_calls reached — stopping instrumented loop.")
            break

    total_elapsed = time.monotonic() - investigation_start

    _header("STEP 4 SUMMARY")
    print(f"Total instrumented investigation wall-clock time: {total_elapsed:.2f}s ({total_elapsed / 60:.1f} min)")
    print(f"Per-round ollama.chat() times (s): {[round(t, 2) for t in round_chat_times]}")
    if round_chat_times:
        avg = sum(round_chat_times) / len(round_chat_times)
        max_t = max(round_chat_times)
        print(f"  average: {avg:.2f}s, max: {max_t:.2f}s ({'outlier present' if max_t > 2 * avg else 'roughly uniform'})")
    if state.final_result:
        print(f"\nfinal_result: {state.final_result}")
    else:
        print("\nNo final_result produced within this diagnostic run's round limit.")


def step5_system_basics() -> None:
    _header("STEP 5: System basics")

    try:
        import psutil

        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"Total RAM: {total_ram_gb:.1f} GiB")
    except ImportError:
        print("psutil not installed — skipping RAM check (install psutil for this, or check manually).")

    print(f"\nOLLAMA_MODEL configured: {_configured_model()}")
    try:
        client = get_client()
        for m in client.list().models:
            if _configured_model() in m.model or m.model in _configured_model():
                print(f"On-disk size per `ollama list`: {m.size / (1024**3):.2f} GiB (digest={m.digest[:12]})")
    except Exception as exc:  # noqa: BLE001
        print(f"Could not query `ollama list` for on-disk size: {exc}")

    log_path = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Ollama", "server.log")
    print(f"\nChecking Ollama server log for GPU/backend detection: {log_path}")
    if os.path.isfile(log_path):
        try:
            with open(log_path, encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
            gpu_lines = [
                line.strip()
                for line in lines
                if any(
                    kw in line
                    for kw in ("gpu memory", "library=", "offloaded", "CPU_Mapped", "n_threads")
                )
            ]
            print(f"Found {len(gpu_lines)} matching log lines. Last 12:")
            for line in gpu_lines[-12:]:
                print(f"  {line}")
        except OSError as exc:
            print(f"Could not read log file: {exc}")
    else:
        print("Log file not found at that path — GPU/backend detection must be checked manually.")


def main() -> None:
    print("Diagnostic run started — no fixes are being applied, this only measures.\n")

    t1 = step1_baseline()
    t2 = step2_with_tools()
    step3_model_residency()
    step4_instrumented_investigation()
    step5_system_basics()

    _header("HEADLINE NUMBERS")
    print(f"Step 1 (baseline, no tools):        {t1:.2f}s")
    print(f"Step 2 (baseline + tool schemas):   {t2:.2f}s  (delta vs step 1: {t2 - t1:+.2f}s)")


if __name__ == "__main__":
    main()
