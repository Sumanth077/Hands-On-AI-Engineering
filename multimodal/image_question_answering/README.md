# Image Question Answering

> Ask visual questions about any page of a PDF document, answered by Gemma 4 with thinking mode.

## Overview

Upload a PDF, select a page, and ask a natural-language question about its visual content. PyMuPDF renders each page into a high-quality image, which is sent alongside your question to Gemma 4 (`gemma-4-31b-it`) via the Gemini API. The model reasons over the rendered image, including charts, tables, diagrams, and figures, and returns a grounded answer in the Gradio interface.

## Demo

![Demo](assets/demo.png)

## Features

- Renders every PDF page to a full-resolution image using PyMuPDF
- Page slider lets you navigate and preview any page before asking
- Answers visual questions grounded in the actual rendered content (charts, tables, figures)
- Gemma 4 thinking mode (`high`) for multi-step reasoning over complex layouts
- Clean two-column Gradio UI with the page viewer on the left and Q&A on the right

## Tech Stack

| Layer | Library |
|---|---|
| PDF to Image | PyMuPDF (`fitz`) |
| LLM | Gemma 4 (`gemma-4-31b-it`) via Gemini API |
| SDK | `google-genai` |
| UI | Gradio |

## Prerequisites

- Python 3.10 or higher
- A Google AI Studio API key with access to Gemma 4

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/Sumanth077/Hands-On-AI-Engineering.git
cd Hands-On-AI-Engineering/multimodal/image_question_answering
```

**2. Create and activate a virtual environment**

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

```bash
cp .env.example .env
```

Open `.env` and fill in your API key (see [Environment Variables](#environment-variables)).

## Usage

```bash
python app.py
```

Gradio will print a local URL (e.g., `http://127.0.0.1:7860`). Open it in your browser, upload a PDF, select a page with the slider, type your question, and click **Answer Question**.

Example questions:
- What does the chart on this page show?
- What is the title of the document?
- List the bullet points in section 2.

## Environment Variables

Create a `.env` file in the project root with the following key:

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Gemini API key; obtain one at [aistudio.google.com](https://aistudio.google.com) |

```env
GOOGLE_API_KEY=your_api_key_here
```

## Project Structure

```text
image-question-answering/
├── app.py          # Gradio app — PDF rendering, Gemma 4 inference, UI
├── .env            # Local secrets (not committed)
├── .env.example    # Environment variable template
└── requirements.txt
```

---

[Back to Top](#image-question-answering)
