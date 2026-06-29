# Offline Medical Agent

> A fully offline agentic RAG system for clinical protocol lookup at remote clinics, field hospitals, and disaster response units. No internet required at runtime.

## Demo

![Demo](assets/demo.gif)

## Overview

Offline Medical Agent is a two-agent pipeline built for medical personnel operating at edge sites with no internet access. A local vector database is pre-loaded with facility-specific clinical protocols covering drug dosages, emergency procedures, and treatment guidelines.

When a clinician describes a patient situation, a **Retrieval Agent** queries the local database using semantic search to find the single most relevant protocol. A **Protocol Agent** then feeds that grounded context to the local language model and generates a concise, actionable response based strictly on the retrieved protocol, without hallucinating outside it.

All inference and retrieval happen entirely on-device.

## Features

- Fully offline after first-run model download
- Two-agent pipeline: semantic retrieval followed by grounded generation
- Ministral 3 3B (Q4_K_M GGUF) via llama-cpp-python, auto-detects GPU (CUDA, Metal) and falls back to CPU
- SentenceTransformers all-MiniLM-L6-v2 for fast local embeddings
- Qdrant local vector database, no server process required
- Five pre-loaded clinical protocols: pediatric fever, anaphylaxis, oral rehydration, wound care, malaria
- Add your own protocols by dropping `.md` or `.txt` files into the `protocols/` folder
- Gradio web UI with example queries

## Prerequisites

- Python 3.9 or higher
- [UV](https://docs.astral.sh/uv/) package manager
- Internet access on first run only (to download the model and embedding weights)

### Platform-specific install for llama-cpp-python

llama-cpp-python must be built with the right backend for your hardware.

**CPU only (any machine)**
```bash
pip install llama-cpp-python
```

**NVIDIA GPU (CUDA)**
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

**Apple Silicon (Metal)**
```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
```

**AMD GPU (ROCm)**
```bash
CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install llama-cpp-python
```

## Setup

```bash
# Clone the repo
git clone https://github.com/Sumanth077/Hands-On-AI-Engineering.git
cd Hands-On-AI-Engineering/ai_agents/offline_medical_agent

# Install llama-cpp-python first using the command above for your hardware, then install the rest
pip install -r requirements.txt

# Run
python app.py
```

On first run the app will:
1. Download Ministral 3 3B Q4_K_M GGUF from HuggingFace (~2 GB) and cache it locally.
2. Download the all-MiniLM-L6-v2 embedding model (~90 MB) and cache it locally.
3. Ingest the five bundled clinical protocols into a local Qdrant database.

Subsequent runs are fully offline.

## Usage

Open the Gradio interface at `http://localhost:7860` and describe a patient situation in plain language. The agent will retrieve the most relevant protocol and generate a grounded clinical response.

To add your own protocols, place `.md` or `.txt` files in the `protocols/` folder and click **Re-load Protocols** in the UI.

## Protocols Directory

The `protocols/` folder is the agent's knowledge base. Every `.md` or `.txt` file in it gets loaded, embedded, and stored in the local Qdrant database when the app starts. When a clinician submits a query, the Retrieval Agent searches this database semantically to find the most relevant protocol, which the Protocol Agent then uses as its sole source of truth.

Five protocols are bundled out of the box:

| File | Contents |
|---|---|
| `pediatric_fever.md` | Fever assessment and antipyretic dosing for children under 5 |
| `anaphylaxis.md` | Adrenaline dosing, airway management, and monitoring for anaphylaxis |
| `oral_rehydration.md` | WHO ORT Plans A, B, and C for dehydration from diarrhoea |
| `wound_care.md` | Wound irrigation, closure, tetanus prophylaxis, and antibiotic guidance |
| `malaria_rdt_treatment.md` | RDT interpretation, AL dosing, and severe malaria emergency management |

### Extending the Agent

The quickest way to make this agent more capable is to add more protocol files. Drop any `.md` or `.txt` file into `protocols/` and click **Re-load Protocols** in the UI. The new content is embedded and searchable immediately, no code changes needed.

Some ideas for what to add:

- Obstetric emergencies (postpartum haemorrhage, eclampsia)
- Neonatal resuscitation
- Burn management
- Snakebite and envenomation
- Sepsis and shock recognition
- Facility-specific drug formulary

The agent has no hardcoded topic limit. The more protocols you add, the better its retrieval coverage. For best results, write each protocol as a single focused document covering one condition or procedure. Mixing multiple unrelated topics in one file will reduce retrieval precision since the entire file gets embedded as a single vector.

## Model

**Ministral 3 3B** (lmstudio-community/Ministral-3-3B-Instruct-2512-GGUF, Q4_K_M)
- 3.4B parameters, ~2 GB on disk
- Native function calling and JSON output
- 256K context window
- Apache 2.0 license
- Runs on any hardware: NVIDIA, Apple Silicon, or CPU-only

## Tech Stack

| Component | Library |
|---|---|
| Local inference | llama-cpp-python |
| Language model | Ministral 3 3B (Q4_K_M GGUF) |
| Embeddings | SentenceTransformers all-MiniLM-L6-v2 |
| Vector database | Qdrant (local file mode) |
| UI | Gradio |

## Disclaimer

This tool is intended to support, not replace, qualified clinical judgment. Always verify responses against official clinical guidelines and apply professional discretion, especially in emergencies.
