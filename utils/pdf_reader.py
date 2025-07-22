import pdfplumber
from typing import List

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
        return '\n'.join(text)
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return str(text)

def chunk_text(text: str, chunk_size: int = 200) -> List[str]:
    """Split text into chunks of approximately chunk_size words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def preview_lines(text: str, n: int = 5) -> list:
    lines = text.splitlines()
    if len(lines) <= 2 * n:
        return lines
    return lines[:n] + ["..."] + lines[-n:]
