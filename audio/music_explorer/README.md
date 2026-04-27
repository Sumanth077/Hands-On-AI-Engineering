# 🎧 Music Explorer

> **Powered by [Gemini 3 Flash](https://ai.google.dev/gemini-api/docs/audio) — ask anything about any audio file or YouTube video**

![Music Explorer Demo](assets/demo.gif)

## Overview

Music Explorer lets you analyze any audio through natural language. Upload a file or paste a YouTube URL, and ask questions — transcribe lyrics, identify instruments, detect emotion, break down track structure, or describe anything you hear. The conversation is multi-turn, so you can keep asking follow-ups without re-uploading.

Gemini 3 Flash handles the audio understanding entirely via the Gemini Files API — no local model, no GPU required.

## Features

- **Multi-turn chat** — ask follow-up questions about the same audio without re-uploading
- **YouTube support** — paste any YouTube URL to download and analyze the audio automatically
- **Five quick-analysis presets** — one click to run full description, transcription, emotion analysis, instrument identification, or track structure breakdown
- **Timestamp-aware responses** — Gemini can reference specific moments in the audio
- **Microphone input** — record directly in the browser for live audio analysis
- **No GPU needed** — runs entirely through the Gemini API

## What You Can Ask

- "Describe this track — genre, tempo, key, instruments, and mood"
- "Transcribe the lyrics with timestamps"
- "What emotion does this convey? Does it shift throughout?"
- "List every instrument and when each one enters"
- "Break down the structure — intro, verse, chorus — with timestamps"
- "What language is being spoken?"
- "Summarize what's being said in this podcast clip"
- "What chord progression is used in the chorus?"

## Tech Stack

| Layer | Technology |
|---|---|
| Model | **Gemini 3 Flash** (`gemini-3-flash-preview`) via Gemini API |
| Audio upload | Gemini Files API |
| YouTube download | yt-dlp + ffmpeg |
| UI | Gradio |
| SDK | google-genai |

## Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- A [Gemini API key](https://aistudio.google.com/app/apikey)
- ffmpeg installed on your system (required for YouTube audio extraction)

**Install ffmpeg:**
```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows
winget install ffmpeg
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Sumanth077/Hands-On-AI-Engineering.git
cd Hands-On-AI-Engineering/audio/audio_intelligence
```

### 2. Set Up Environment Variables

```bash
cp .env.example .env
```

Add your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Get your key at [aistudio.google.com](https://aistudio.google.com/app/apikey).

### 3. Install Dependencies

```bash
uv sync
```

## Usage

```bash
uv run python app.py
```

Navigate to `http://localhost:7860`.

1. **Upload an audio file** or paste a YouTube URL and click **Load YouTube Audio**
2. **Type a question** or click a **Quick Analysis** preset
3. **Keep asking follow-ups** — the conversation is multi-turn, no need to re-upload

## How It Works

```
Audio file / YouTube URL
         │
         ▼
  yt-dlp (YouTube only)    ← downloads audio as MP3
         │
         ▼
  Gemini Files API          ← uploads audio, returns a file reference
         │
         ▼
  Gemini 3 Flash            ← multi-turn conversation with audio context
         │
         ▼
  Streamed response         ← transcription · analysis · timestamps · description
```

## Project Structure

```text
audio_intelligence/
├── app.py            # Gradio UI and Gemini inference logic
├── assets/
│   └── demo.png      # App screenshot
├── .env.example      # API key template
├── pyproject.toml    # Dependencies (uv)
└── README.md
```

## Notes

- Uploaded files are stored temporarily in the Gemini Files API for up to 48 hours
- YouTube audio is cached locally for the session to avoid repeat downloads
- For best results, use audio files under 20 MB

## License

This project is licensed under the MIT License. See the [LICENSE](../../LICENSE) file for details.

---

[Back to Top](#-music-explorer)
