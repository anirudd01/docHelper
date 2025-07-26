import os
from typing import List, Optional

import numpy as np
import pdfplumber
import re
from utils.text_cleaner import TextCleaner

# Always resolve storage relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_STORAGE_DIR = os.path.join(PROJECT_ROOT, "data", "pdfs")
VECTOR_STORAGE_DIR = os.path.join(PROJECT_ROOT, "data", "vectors")
os.makedirs(PDF_STORAGE_DIR, exist_ok=True)
os.makedirs(VECTOR_STORAGE_DIR, exist_ok=True)

TEXT_STORAGE_DIR = os.path.join(PROJECT_ROOT, "data", "texts")
os.makedirs(TEXT_STORAGE_DIR, exist_ok=True)

CLEANED_TEXT_STORAGE_DIR = os.path.join(PROJECT_ROOT, "data", "cleaned")
os.makedirs(CLEANED_TEXT_STORAGE_DIR, exist_ok=True)


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


def store_cleaned_text(text: str, filename: str) -> str:
    """Save cleaned text to the cleaned text storage directory as .txt."""
    file_path = os.path.join(CLEANED_TEXT_STORAGE_DIR, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    return file_path


def fetch_cleaned_text(filename: str) -> Optional[str]:
    """Get the path to a stored cleaned text file if it exists."""
    file_path = os.path.join(CLEANED_TEXT_STORAGE_DIR, filename)
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


def advanced_chunk_text(
    text: str, chunk_size: int = 200, overlap: int = 30
) -> List[str]:
    """
    Split cleaned text into sentence-based chunks with overlap for better context.
    Args:
        text (str): Cleaned text to chunk.
        chunk_size (int): Target number of words per chunk.
        overlap (int): Number of words to overlap between chunks.
    Returns:
        List[str]: List of text chunks.
    """
    # Clean the text first (in case not already cleaned)
    text = TextCleaner.clean_text(text)
    # Split into sentences (simple regex, can be improved)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk = []
    current_len = 0
    for sent in sentences:
        words = sent.split()
        if current_len + len(words) > chunk_size and current_chunk:
            # Save current chunk
            chunks.append(" ".join(current_chunk))
            # Start new chunk with overlap
            if overlap > 0:
                overlap_words = (
                    current_chunk[-overlap:]
                    if len(current_chunk) >= overlap
                    else current_chunk
                )
                current_chunk = overlap_words.copy()
                current_len = len(current_chunk)
            else:
                current_chunk = []
                current_len = 0
        current_chunk.extend(words)
        current_len += len(words)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return [c.strip() for c in chunks if c.strip()]


def read_cleaned_chunks_and_vectors(
    pdf_filename: str, chunk_size: int = 200, overlap: int = 30
) -> Optional[tuple]:
    """Return (chunks, vectors) for a given PDF filename using cleaned text and advanced chunking, or None if not available."""
    fname_without_ext = os.path.splitext(pdf_filename)[0]
    # Cleaned text file
    cleaned_text_filename = fname_without_ext + ".txt"
    cleaned_text_path = fetch_cleaned_text(cleaned_text_filename)
    if not cleaned_text_path:
        return None
    with open(cleaned_text_path, "r", encoding="utf-8") as f:
        cleaned_text = f.read()
    chunks = advanced_chunk_text(cleaned_text, chunk_size=chunk_size, overlap=overlap)
    # Vector file
    vector_filename = fname_without_ext + ".npy"
    vector_path = fetch_vectors(vector_filename)
    if not vector_path:
        return None
    vectors = np.load(vector_path)
    if len(chunks) != len(vectors):
        return None
    return (chunks, vectors)


def remove_pdf_files(filename: str) -> dict:
    """
    Remove PDF-related files from the file system.

    Args:
        filename: Name of the PDF file
        keep_pdf: If True, keep the original PDF file. If False, remove it too.

    Returns:
        dict: Status of the operation with list of removed files
    """
    removed_files = []
    errors = []

    # Remove .pdf extension for text/vector file names
    base_name = filename.replace(".pdf", "")

    text_path = fetch_text(f"{base_name}.txt")
    if text_path and os.path.exists(text_path):
        try:
            os.remove(text_path)
            removed_files.append(f"text: {text_path}")
        except Exception as e:
            errors.append(f"Failed to remove text file: {e}")
    else:
        errors.append("Text file not found: text_path")

    # Remove cleaned text file
    cleaned_text_path = fetch_cleaned_text(f"{base_name}.txt")
    if cleaned_text_path and os.path.exists(cleaned_text_path):
        try:
            os.remove(cleaned_text_path)
            removed_files.append(f"cleaned text: {cleaned_text_path}")
        except Exception as e:
            errors.append(f"Failed to remove cleaned text file: {e}")
    else:
        errors.append("Cleaned text file not found: cleaned_text_path")

    # Remove vector file
    vector_path = fetch_vectors(f"{base_name}.npy")
    if vector_path and os.path.exists(vector_path):
        try:
            os.remove(vector_path)
            removed_files.append(f"vectors: {vector_path}")
        except Exception as e:
            errors.append(f"Failed to remove vector file: {e}")
    else:
        errors.append("Vector file not found: vector_path")

    return {
        "success": len(errors) == 0,
        "removed_files": removed_files,
        "errors": errors,
        "message": f"Removed {len(removed_files)} files for '{filename}'",
    }
