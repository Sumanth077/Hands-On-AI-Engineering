# Medical Prescription Digitizer

> Extract, structure, and validate handwritten or printed prescriptions using Mistral Large 3 and RxNorm.

<<<<<<< HEAD
## Overview

Upload a prescription image — handwritten or printed — and the app uses Mistral Large 3's vision capabilities to read and interpret it, including messy handwriting and medical abbreviations. Extracted drug names are then validated against the [RxNorm](https://rxnav.nlm.nih.gov/) drug database (free, no API key required). Validated drugs are highlighted in green; unrecognised or flagged ones appear in red.
=======
## Demo

![Demo](assets/demo.gif)

## Overview

Upload a prescription image (handwritten or printed) and the app uses Mistral Large 3's vision capabilities to read and interpret it, including messy handwriting and medical abbreviations. Extracted drug names are then validated against the [RxNorm](https://rxnav.nlm.nih.gov/) drug database (free, no API key required). Validated drugs are highlighted in green; unrecognised or flagged ones appear in red.
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a

## Features

- Reads handwritten and printed prescriptions via multimodal LLM
- Decodes standard medical abbreviations (QD, BID, TID, PRN, PO, etc.)
- Extracts patient name, doctor name, date, medications with dosage and frequency, and additional notes
- Flags illegible or uncertain fields instead of guessing
- Validates every drug name against the RxNorm API in real time
- Displays structured results with per-medication validation status and RxNorm IDs

## Tech Stack

| Layer | Technology |
|---|---|
| Model | Mistral Large 3 (`mistral-large-latest`) |
| Structured outputs | `mistralai` SDK `chat.parse` + Pydantic v2 |
| Schema validation | Pydantic v2 |
| Drug validation | RxNorm API (free, no key needed) |
| UI | Streamlit |

## Prerequisites

- Python 3.10 or higher
<<<<<<< HEAD
- A Mistral API key — get one at [platform.mistral.ai](https://platform.mistral.ai)
=======
- A Mistral API key. Get one at [platform.mistral.ai](https://platform.mistral.ai)
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a

## Installation

```bash
git clone https://github.com/Sumanth077/Hands-On-AI-Engineering.git
<<<<<<< HEAD
cd Hands-On-AI-Engineering/ocr/medical-prescription-digitizer
=======
cd Hands-On-AI-Engineering/OCR/medical_prescription_digitizer
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a
```

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the example env file and add your API key:

```bash
cp .env.example .env
```

## Usage

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser, enter your Mistral API key in the sidebar, upload a prescription image (JPG, PNG, or WEBP), then click **Digitize Prescription**.

**Example input:** a handwritten prescription image containing:

```text
Patient: John Smith          Date: 04/18/2026
Dr. Sarah Lee, MD

Rx:
<<<<<<< HEAD
1. Amoxicillin 500mg — TID x 7 days
2. Ibuprofen 400mg  — BID PRN pain
3. [illegible] 10mg — QD
=======
1. Amoxicillin 500mg, TID x 7 days
2. Ibuprofen 400mg, BID PRN pain
3. [illegible] 10mg, QD
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a
```

**Structured output:**

```text
Patient      John Smith
Doctor       Dr. Sarah Lee
Date         04/18/2026

Medications
✅ Amoxicillin   500 mg · three times daily · 7 days   [RxNorm ID: 723]
✅ Ibuprofen     400 mg · twice daily · as needed       [RxNorm ID: 5640]
<<<<<<< HEAD
⚠️ [illegible]  10 mg  · once daily                   Drug name not found in RxNorm — possible misread
=======
⚠️ [illegible]  10 mg  · once daily                   Drug name not found in RxNorm, possible misread
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a

Illegible fields
• Medication 3 drug name unclear
```

## Environment Variables

| Variable | Description |
|---|---|
| `MISTRAL_API_KEY` | Your Mistral API key from [platform.mistral.ai](https://platform.mistral.ai) |

## Project Structure

```text
medical-prescription-digitizer/
<<<<<<< HEAD
├── app.py            # Streamlit UI — upload, display, and orchestration
=======
├── app.py            # Streamlit UI: upload, display, and orchestration
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a
├── extractor.py      # Sends image to Mistral Large 3 and returns a Prescription object
├── schemas.py        # Pydantic models: Prescription and Medication
├── validator.py      # Queries RxNorm API to validate each drug name
├── requirements.txt  # Python dependencies
└── .env.example      # Environment variable template
```
<<<<<<< HEAD
=======

---

[Back to top](#medical-prescription-digitizer)
>>>>>>> 1d1e9f137cfd1123edbae5d8e955ce0b9c7fcf4a
