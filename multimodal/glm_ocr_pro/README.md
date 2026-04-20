# GLM-OCR Pro 🧾

**GLM-OCR Pro** is a high-performance, local-first Streamlit application designed for structured document extraction. It leverages the **GLM-OCR** model via Ollama to transform images and PDFs into cleanly formatted Markdown in real-time.

---
## Overview
GLM-OCR Pro solves the problem of extracting structured text from documents without sending sensitive data to external APIs. It runs entirely on your local machine using the GLM-OCR model via Ollama, making it ideal for confidential documents. The app streams results in real-time and outputs clean, structured Markdown that preserves headings, tables, and lists.

## Prerequisites
Before you begin, ensure you have:
- Python 3.8 or higher
- [Ollama](https://ollama.com) installed and running
- The GLM-OCR model pulled: `ollama pull glm-ocr`
- A GPU with at least 4GB VRAM (recommended)

## ✨ Features

- **100% Private & Local:** No data ever leaves your machine. Perfect for sensitive documents.
- **Real-time Streaming:** Watch text populate your screen as the model identifies it.
- **Premium UI:** A modern, glassmorphism-inspired dashboard with support for complex tables.
- **Hardware Optimized:** Pre-configured performance profiles tailored for laptop and desktop GPUs.
- **Multi-Format Support:** Seamlessly handles PNG, JPG, JPEG, and multi-page PDFs.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Sumanth077/Hands-On-AI-Engineering.git
cd Hands-On-AI-Engineering/multimodal_apps/glm-ocr-pro
```

### 2. Install Ollama
Download and install Ollama from [ollama.com](https://ollama.com).

### 3. Pull the GLM-OCR Model
Open your terminal and run:
```bash
ollama pull glm-ocr
```

### 4. Setup the Python Environment
```bash
# Optional: Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Or `.venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

### 5. Run the App
```bash
streamlit run app.py
```

---

## ⚙️ Performance Tuning

GLM-OCR Pro includes three "Performance Profiles" to help you balance speed and accuracy based on your hardware:

| Profile | Image Res. | Grayscale | Use Case |
| :--- | :--- | :--- | :--- |
| **Balanced** | 1300px | No | High-accuracy scanning on desktop GPUs. |
| **Fast** | 1000px | Yes | Optimal for mid-range GPUs (e.g., 4GB VRAM). |
| **Ultra Fast** | 800px | Yes | Lightning-fast extractions for simple text. |

---

## 🛠️ Tech Stack

- **UI Framework:** Streamlit
- **PDF Engine:** PyPDFium2
- **Image Processing:** Pillow
- **Inference:** Ollama (`glm-ocr`)

---

## 📝 License

This project is intended for personal and professional use. Ensure compliance with the GLM-OCR and Ollama licensing terms.
