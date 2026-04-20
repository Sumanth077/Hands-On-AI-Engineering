# LaTeX Formula OCR — Extract and render math formulas from images using local AI

## Overview

Reading mathematical formulas from images or scanned documents is tedious and error-prone when done by hand. This tool solves that by running a vision-language model entirely on your machine — no cloud services, no API keys — to detect every equation in an uploaded image or PDF and instantly render it in a polished, browser-native display.

It works by sending the image to **GLM-OCR** (served locally via Ollama), which returns raw LaTeX source for each detected formula. The app then parses those results and renders them visually using **KaTeX**, all within a Streamlit web interface.

Students, researchers, engineers, and anyone who regularly works with mathematical documents will benefit from this tool — particularly when digitising textbooks, lecture slides, or handwritten notes.

## Features

- **Fully local inference** — GLM-OCR runs via Ollama; no API keys or internet access required for OCR
- **Image and PDF support** — upload PNG, JPG/JPEG, or single-page PDFs (first page rendered automatically)
- **Multi-formula extraction** — detects and extracts every equation in the image in one pass
- **In-browser KaTeX rendering** — formulas are rendered in display mode with proper mathematical typesetting
- **LaTeX source viewer** — each formula card shows the raw LaTeX alongside the rendered output
- **One-click copy** — copy any formula's LaTeX source to the clipboard instantly
- **Live Ollama status** — sidebar shows whether Ollama is running and whether glm-ocr is installed
- **Auto image resizing** — large images are downscaled before inference to keep requests fast

## Tech Stack

### Models & Frameworks

| Component | Details |
|-----------|---------|
| **GLM-OCR** | Vision-language model for mathematical formula recognition, served via Ollama |
| **Streamlit** | Python web framework powering the interactive UI |
| **KaTeX 0.16.11** | Browser-side LaTeX rendering engine (loaded from jsDelivr CDN) |

### Libraries

| Library | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `requests` | HTTP client for communicating with the Ollama local API |
| `Pillow` | Image loading and resizing |
| `PyMuPDF` (`fitz`) | PDF-to-image conversion |

## Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.ai/download)** installed and running
- **glm-ocr** model pulled into Ollama:
  ```bash
  ollama pull glm-ocr
  ```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Sumanth077/Hands-On-AI-Engineering.git
cd Hands-On-AI-Engineering/OCR/latex_formula_ocr
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

Make sure Ollama is running, then launch the app:

```bash
ollama serve          # start Ollama if not already running
streamlit run app.py
```

Open your browser at `http://localhost:8501`, upload an image or PDF containing mathematical formulas, and click **Extract Formulas**. Each detected formula is displayed as a rendered equation alongside its copyable LaTeX source.

**Example workflow:**

```bash
# 1. Pull the model (one-time setup)
ollama pull glm-ocr

# 2. Start Ollama
ollama serve

# 3. Launch the app
streamlit run app.py
```

## Project Structure

```text
latex-formula-ocr/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```
