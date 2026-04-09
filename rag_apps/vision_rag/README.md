# VisionRAG: Multimodal Search & Visual Question Answering

A powerful Streamlit application that combines multimodal search with visual question answering to analyze images and PDF documents using AI.

## Features

- **Multimodal Search**: Uses Cohere Embed-4 to find semantically relevant images for text questions
- **Visual Question Answering**: Employs Google Gemini 2.5 Flash to analyze images and generate context-aware answers
- **Flexible Content Sources**:
  - Upload custom images (PNG, JPG, JPEG)
  - Upload PDF documents (automatically extracts pages as images)
  - Load sample financial charts and infographics
- **No OCR Required**: Directly processes complex visual elements without text extraction

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Cohere API key
- Google Gemini API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd vision-rag
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API keys**
   - Get your [Cohere API key](https://cohere.ai/)
   - Get your [Google Gemini API key](https://makersuite.google.com/app/apikey)
   - Create a `.env` file in the project root
   - Add your API keys to the `.env` file:
     ```
     COHERE_API_KEY=your_cohere_api_key_here
     GEMINI_API_KEY=your_gemini_api_key_here
     ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Dependencies

- **streamlit**: Web application framework
- **cohere**: Embedding generation
- **requests**: HTTP requests for Gemini API
- **PyMuPDF**: PDF processing
- **Pillow**: Image processing
- **numpy**: Numerical operations
- **scikit-learn**: Cosine similarity calculations


## Contributing

Contributions, issues, and feature requests are welcome. Please feel free to submit a Pull Request or open an issue.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

