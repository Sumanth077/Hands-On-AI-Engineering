import os
import tempfile
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader


class DocumentProcessor:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def process(self, file_content: bytes, filename: str) -> list:
        suffix = Path(filename).suffix.lower()
        if suffix not in {".pdf", ".txt"}:
            raise ValueError(f"Unsupported file type: {suffix}. Use PDF or TXT.")

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path, encoding="utf-8")

            docs = loader.load()
            chunks = self.splitter.split_documents(docs)
            return chunks
        finally:
            os.unlink(tmp_path)
