# 📝 Form-Fill AI: Pro Workspace

A powerful, agentic form-filling application that uses **Landing AI** for Layout Parsing and **MiniMax M2.7** for multi-turn conversational data gathering. Features a cinematic "Live Inking" engine for visual document stamping and a full dual-pane audit workflow.

---

## 🎨 Key Features

*   **9-Step "Agentic" Workflow**: A structured journey from Document Analysis to Final Export.
*   **Conversational Data Gathering**: Talk to the Agent to provide missing information via a persistent chat interface.
*   **Flicker-Free Live Auto-Filling**: Watch the document update in real-time as the Agent "inks" the fields without browser flickering.
*   **Dual-Pane Verification**: Compare the original blank form and the final result side-by-side.
*   **Session Persistence**: Every workspace is saved locally. Refreshing your browser or restarting the app will restore your progress perfectly via URL-based Session IDs.
*   **History Explorer**: A sidebar gallery to manage and toggle between multiple active forms.

---

## 🏎️ The 9-Step Journey

1.  **Upload**: Provide the target form (PDF/Image) and optional source context (CV/ID).
2.  **Analysis**: Landing AI parses the physical layout and detects interactive fields.
3.  **Extraction**: The Agent extracts values from your source documents.
4.  **Audit**: The Agent reports what was found and what is still missing.
5.  **Conversation**: Fill in missing gaps or clarify information via the chat interface.
6.  **The Trigger**: Once details are complete, the user triggers the "Visual Auto-Fill".
7.  **Auto-Filling**: A cinematic live-action loop "paints" the PDF field-by-field.
8.  **Verification**: A high-resolution side-by-side audit of the Original vs. Filled PDF.
9.  **Export**: Download the finalized PDF or the raw JSON data registry.

---

## 🛠️ Setup & Installation

### 1. Prerequisites
You will need API keys for the following services:
*   **Landing AI**: For high-accuracy Document extraction.
*   **MiniMax**: For the M2.7 LLM conversation and logic.

### 2. Environment Configuration
Create a `.env` file in the root directory:
```env
VISION_AGENT_API_KEY=your_landing_ai_key
MINIMAX_API_KEY=your_minimax_key
```

### 3. Clone the Repository
```bash
git clone https://github.com/Sumanth077/Hands-On-AI-Engineering.git
cd Hands-On-AI-Engineering/ai_agents/agentic-form-filler
```

### 4. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 How to Run

1.  Start the application using the provided batch script (Windows):
    ```cmd
    run.cmd
    ```
    *Or run manually via:*
    ```bash
    streamlit run app.py
    ```
2.  Upload the provided `sample_complex_employment_application.pdf` to see the Agent in action!

---

## 🏗️ Technology Stack

*   **Frontend**: Streamlit (Python-based Web Framework)
*   **Layout Engine**: Landing AI (Vision Agent API)
*   **Intelligence**: MiniMax M2.7 (Large Language Model)
*   **Document Engine**: PyMuPDF (Fitz)
*   **Persistence**: URL-based State Synchronization & JSON Storage
