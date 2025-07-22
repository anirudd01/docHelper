import os
from typing import List, Optional

import numpy as np
import pdfplumber

# Always resolve storage relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_STORAGE_DIR = os.path.join(PROJECT_ROOT, "data", "pdfs")
VECTOR_STORAGE_DIR = os.path.join(PROJECT_ROOT, "data", "vectors")
os.makedirs(PDF_STORAGE_DIR, exist_ok=True)
os.makedirs(VECTOR_STORAGE_DIR, exist_ok=True)

TEXT_STORAGE_DIR = os.path.join(PROJECT_ROOT, "data", "texts")
os.makedirs(TEXT_STORAGE_DIR, exist_ok=True)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract pure text from a PDF file, ignoring images and formatting."""
    try:
        text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
                text.append("\n")
        return "\n".join(text)
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return str(text)


def chunk_text(text: str, chunk_size: int = 200) -> List[str]:
    """Split text into chunks of approximately chunk_size words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


def preview_lines(text: str, n: int = 5) -> list:
    lines = text.splitlines()
    if len(lines) <= 2 * n:
        return lines
    return lines[:n] + ["..."] + lines[-n:]


def list_pdfs() -> List[str]:
    """List all PDF files in the storage directory."""
    return [f for f in os.listdir(PDF_STORAGE_DIR) if f.lower().endswith(".pdf")]


def search_pdfs(query: str) -> List[str]:
    """Search for PDF files by filename substring (case-insensitive)."""
    return [f for f in list_pdfs() if query.lower() in f.lower()]


def fetch_pdf(filename: str) -> Optional[str]:
    """Get the path to a stored PDF file if it exists."""
    file_path = os.path.join(PDF_STORAGE_DIR, filename)
    return file_path if os.path.exists(file_path) else None


def store_pdf(file_bytes: bytes, filename: str) -> str:
    """Save a PDF file to the storage directory."""
    file_path = os.path.join(PDF_STORAGE_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    return file_path


def store_vectors(vectors, filename: str) -> str:
    """Save a vector file (e.g., .npy) to the vector storage directory."""
    file_path = os.path.join(VECTOR_STORAGE_DIR, filename)
    import numpy as np

    np.save(file_path, vectors)
    return file_path


def fetch_vectors(filename: str) -> Optional[str]:
    """Get the path to a stored vector file if it exists."""
    file_path = os.path.join(VECTOR_STORAGE_DIR, filename)
    return file_path if os.path.exists(file_path) else None


def list_vectors() -> List[str]:
    """List all vector files in the storage directory."""
    return [f for f in os.listdir(VECTOR_STORAGE_DIR) if f.lower().endswith(".npy")]


def store_text(text: str, filename: str) -> str:
    """Save extracted text to the text storage directory as .txt."""
    file_path = os.path.join(TEXT_STORAGE_DIR, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    return file_path


def fetch_text(filename: str) -> Optional[str]:
    """Get the path to a stored text file if it exists."""
    file_path = os.path.join(TEXT_STORAGE_DIR, filename)
    return file_path if os.path.exists(file_path) else None


def read_chunks_and_vectors(
    pdf_filename: str, chunk_size: int = 200
) -> Optional[tuple]:
    """Return (chunks, vectors) for a given PDF filename, or None if not available."""
    fname_without_ext = os.path.splitext(pdf_filename)[0]
    # Text file
    text_filename = fname_without_ext + ".txt"
    text_path = fetch_text(text_filename)
    if not text_path:
        return None
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text, chunk_size=chunk_size)
    # Vector file
    vector_filename = fname_without_ext + ".npy"
    vector_path = fetch_vectors(vector_filename)
    if not vector_path:
        return None
    vectors = np.load(vector_path)
    if len(chunks) != len(vectors):
        return None
    return (chunks, vectors)
