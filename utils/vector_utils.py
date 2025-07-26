import os
from typing import List, Optional

import numpy as np
from fastembed import TextEmbedding

# from sentence_transformers import SentenceTransformer  # Commented out - replaced with FastEmbed
from utils.db_utils import (
    get_or_create_org,
    insert_chunk,
    insert_embedding,
    insert_pdf,
)
from utils.file_manager import (
    advanced_chunk_text,
    extract_text_from_pdf,
    store_cleaned_text,
    store_text,
    store_vectors,
)
from utils.text_cleaner import TextCleaner

# Default embedding model and size (can be changed later)
# OLD: DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # SentenceTransformer model
DEFAULT_EMBEDDING_MODEL = (
    "BAAI/bge-small-en-v1.5"  # FastEmbed model - lightweight and effective
)
SAVE_TO_DB = True


# OLD SentenceTransformer Implementation (commented out)
# class Embedder:
#     def __init__(self, model_name: Optional[str] = None):
#         self.model_name = model_name or DEFAULT_EMBEDDING_MODEL
#         self.model = SentenceTransformer(self.model_name)
#
#     def embed(self, texts: List[str]) -> List[List[float]]:
#         return self.model.encode(texts, show_progress_bar=False).tolist()


# NEW FastEmbed Implementation
class Embedder:
    def __init__(self, model_name: Optional[str] = None):
        try:
            self.model_name = model_name or DEFAULT_EMBEDDING_MODEL
            self.model = TextEmbedding(model_name=self.model_name)
            print(f"FastEmbed model '{self.model_name}' loaded successfully")
        except ImportError:
            raise ImportError(
                "FastEmbed not installed. Install with: pip install fastembed"
            )

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using FastEmbed.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors as lists of floats
        """
        # FastEmbed returns a generator, so we convert to list
        embeddings = list(self.model.embed(texts))
        # Convert numpy arrays to lists of floats for compatibility
        return [embedding.tolist() for embedding in embeddings]


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

    # Clean text using enhanced cleaner and store cleaned version
    cleaned_text = TextCleaner.clean_text_aggressive(extracted_text)
    cleaned_text_filename = fname_without_ext + ".txt"
    store_cleaned_text(cleaned_text, cleaned_text_filename)

    # Advanced chunking on cleaned text
    chunks = advanced_chunk_text(cleaned_text, chunk_size=chunk_size)

    # Additional cleaning of individual chunks to ensure no artifacts remain
    cleaned_chunks = []
    for chunk in chunks:
        # Apply additional cleaning to each chunk
        cleaned_chunk = TextCleaner.clean_text_aggressive(chunk)
        if cleaned_chunk.strip():  # Only keep non-empty chunks
            cleaned_chunks.append(cleaned_chunk)

    # Embed and store vectors using cleaned chunks
    embedder = get_embedder(model_name)
    vectors = embedder.embed(cleaned_chunks)
    vector_filename = fname_without_ext + ".npy"
    store_vectors(vectors, vector_filename)

    print(
        f"Saved raw text to {text_filename}, cleaned text to {cleaned_text_filename}, and vectors to {vector_filename}"
    )
    print(
        f"Generated {len(cleaned_chunks)} cleaned chunks from {len(chunks)} original chunks"
    )

    if SAVE_TO_DB:
        save_in_db(pdf_filename, chunk_size, cleaned_chunks, vectors)


def save_in_db(pdf_filename, chunk_size, chunks, vectors):
    print(f"Saving {pdf_filename} to db")
    org_id = get_or_create_org("default")
    pdf_id = insert_pdf(org_id, pdf_filename, chunk_size)

    # Ensure we're saving cleaned chunks
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        # Final safety check - clean the chunk one more time before saving
        final_chunk = TextCleaner.clean_text_aggressive(chunk)
        if final_chunk.strip():  # Only save non-empty chunks
            chunk_id = insert_chunk(pdf_id, idx, final_chunk)
            insert_embedding(chunk_id, vector)

    print(f"Saved {pdf_filename} to db with {len(chunks)} cleaned chunks")


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
