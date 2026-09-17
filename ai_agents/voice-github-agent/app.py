"""Voice GitHub Agent — web app entry point.

Flow:
  1. Browser records a spoken instruction and encodes it as WAV client-side.
  2. POST /api/run receives that WAV, transcribes it with AssemblyAI's Sync
     API, hands the transcript to the tool-calling agent, and returns the
     transcript, the tool calls made, and the agent's final summary as JSON.

Run: python app.py, then open http://localhost:5000
"""
import os
import tempfile

from dotenv import find_dotenv, load_dotenv

# load_dotenv() does NOT override a variable that's already set in the real
# OS environment — if GITHUB_REPO/GITHUB_TOKEN were ever `export`ed (or set
# in a parent shell/terminal) in this session, that value silently wins over
# .env every time, regardless of how many times .env is edited or the app is
# restarted. If GITHUB_REPO ever looks stale, check `echo $GITHUB_REPO` /
# `$env:GITHUB_REPO` in the shell you're launching `python app.py` from
# before assuming .env itself is wrong.
load_dotenv()

print(f"[startup] .env resolved to: {find_dotenv() or '(none found)'}")
print(f"[startup] GITHUB_REPO = {os.environ.get('GITHUB_REPO')!r}")

from flask import Flask, jsonify, render_template, request

from agent import run_agent
from transcribe import transcribe_audio

REQUIRED_ENV_VARS = ("ASSEMBLYAI_API_KEY", "GITHUB_TOKEN", "GITHUB_REPO")

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/run", methods=["POST"])
def api_run():
    missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        return jsonify({"error": f"Missing env var(s): {', '.join(missing)}. Check your .env file."}), 500

    audio_file = request.files.get("audio")
    if audio_file is None:
        return jsonify({"error": "No audio file uploaded."}), 400

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "instruction.wav")
        audio_file.save(wav_path)

        try:
            transcription = transcribe_audio(wav_path)
        except Exception as exc:
            return jsonify({"error": f"Transcription failed: {exc}"}), 502

    transcript = (transcription.get("text") or "").strip()
    if not transcript:
        return jsonify({"error": "Got an empty transcript — try again and speak a bit longer."}), 400

    try:
        agent_result = run_agent(transcript)
    except Exception as exc:
        return jsonify({"error": f"Agent failed: {exc}", "transcript": transcript}), 502

    return jsonify(
        {
            "transcript": transcript,
            "tool_calls": agent_result["tool_calls"],
            "summary": agent_result["summary"],
        }
    )


if __name__ == "__main__":
    missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        raise SystemExit(
            f"Missing env var(s): {', '.join(missing)}. Copy .env.example to .env and fill it in."
        )
    app.run(debug=True, port=5000)
