import os
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from utils.db_utils import (
    get_or_create_org,
    insert_chunk,
    insert_embedding,
    insert_pdf,
)
from utils.file_manager import (
    extract_text_from_pdf,
    store_text,
    store_vectors,
    store_cleaned_text,
    advanced_chunk_text,
)
from utils.text_cleaner import TextCleaner

# Default embedding model and size (can be changed later)
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SAVE_TO_DB = True


class Embedder:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or DEFAULT_EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()


def get_embedder(model_name: Optional[str] = None):
    global EMBEDDER, EMBEDDING_MODEL
    if (
        "EMBEDDER" not in globals()
        or EMBEDDER is None
        or (model_name and model_name != EMBEDDING_MODEL)
    ):
        globals()["EMBEDDER"] = Embedder(model_name=model_name)
        globals()["EMBEDDING_MODEL"] = globals()["EMBEDDER"].model_name
    return globals()["EMBEDDER"]


def extract_embed_n_save(pdf_path, chunk_size, model_name, pdf_filename):
    # Store raw text
    extracted_text = extract_text_from_pdf(pdf_path)
    fname_without_ext = os.path.splitext(pdf_filename)[0]
    text_filename = fname_without_ext + ".txt"
    store_text(extracted_text, text_filename)
    # Clean text and store cleaned version
    cleaned_text = TextCleaner.clean_text(extracted_text)
    cleaned_text_filename = fname_without_ext + ".txt"
    store_cleaned_text(cleaned_text, cleaned_text_filename)
    # Advanced chunking on cleaned text
    chunks = advanced_chunk_text(cleaned_text, chunk_size=chunk_size)
    # Embed and store vectors
    embedder = get_embedder(model_name)
    vectors = embedder.embed(chunks)
    vector_filename = fname_without_ext + ".npy"
    store_vectors(vectors, vector_filename)
    print(
        f"Saved raw text to {text_filename}, cleaned text to {cleaned_text_filename}, and vectors to {vector_filename}"
    )
    if SAVE_TO_DB:
        save_in_db(pdf_filename, chunk_size, chunks, vectors)


def save_in_db(pdf_filename, chunk_size, chunks, vectors):
    print(f"Saving {pdf_filename} to db")
    org_id = get_or_create_org("default")
    pdf_id = insert_pdf(org_id, pdf_filename, chunk_size)
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        chunk_id = insert_chunk(pdf_id, idx, chunk)
        insert_embedding(chunk_id, vector)
    print(f"Saved {pdf_filename} to db")


def cosine_sim(a: List[float], b: List[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def get_top_k_indices(scores: List[float], k: int) -> List[int]:
    arr = np.array(scores)
    return arr.argsort()[-k:][::-1].tolist()


# Usage:
# embedder = Embedder()
# vectors = embedder.embed(["chunk1", "chunk2"])
