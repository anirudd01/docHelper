import os
from typing import List, Optional

PDF_STORAGE_DIR = os.path.join(os.path.dirname(__file__), 'data', 'pdfs')
print(PDF_STORAGE_DIR)
os.makedirs(PDF_STORAGE_DIR, exist_ok=True)

def store_pdf(file_bytes: bytes, filename: str) -> str:
    """Save a PDF file to the storage directory."""
    file_path = os.path.join(PDF_STORAGE_DIR, filename)
    with open(file_path, 'wb') as f:
        f.write(file_bytes)
    return file_path

def fetch_pdf(filename: str) -> Optional[str]:
    """Get the path to a stored PDF file if it exists."""
    file_path = os.path.join(PDF_STORAGE_DIR, filename)
    return file_path if os.path.exists(file_path) else None

def list_pdfs() -> List[str]:
    """List all PDF files in the storage directory."""
    return [f for f in os.listdir(PDF_STORAGE_DIR) if f.lower().endswith('.pdf')]

def search_pdfs(query: str) -> List[str]:
    """Search for PDF files by filename substring (case-insensitive)."""
    return [f for f in list_pdfs() if query.lower() in f.lower()] 