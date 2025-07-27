import concurrent.futures
import os
import time
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from utils import timeit
from utils.db_utils import (
    bulk_insert_chunks_with_embeddings_copy,
    get_or_create_org,
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
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SAVE_TO_DB = True


# OLD SentenceTransformer Implementation (commented out)
# class Embedder:
#     def __init__(self, model_name: Optional[str] = None):
#         self.model_name = model_name or DEFAULT_EMBEDDING_MODEL
#         self.model = SentenceTransformer(self.model_name)
#
#     def embed(self, texts: List[str]) -> List[List[float]]:
#         return self.model.encode(texts, show_progress_bar=False).tolist()


# NEW SentenceTransformer Implementation with Async Processing
class Embedder:
    @timeit
    def __init__(self, model_name: Optional[str] = None):
        try:
            self.model_name = model_name or DEFAULT_EMBEDDING_MODEL
            self.model = SentenceTransformer(self.model_name)
            print(f"SentenceTransformer model '{self.model_name}' loaded successfully")
        except ImportError:
            raise ImportError(
                "SentenceTransformers not installed. Install with: pip install sentence-transformers"
            )

    @timeit
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using SentenceTransformers.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors as lists of floats
        """
        # SentenceTransformers returns numpy arrays, convert to lists
        embeddings = self.model.encode(texts, show_progress_bar=False)
        # Convert numpy arrays to lists of floats for compatibility
        return [embedding.tolist() for embedding in embeddings]

    @timeit
    def embed_async(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings asynchronously with smart batching.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors as lists of floats
        """
        # Small inputs: use single thread (no overhead)
        if len(texts) < 10:
            return self.embed(texts)

        # Large inputs: use parallel processing with threads (can share model)
        batch_size = max(50, len(texts) // (os.cpu_count() or 4))
        print(
            f"Processing {len(texts)} texts in batches of {batch_size} using parallel processing"
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            batches = [
                texts[i : i + batch_size] for i in range(0, len(texts), batch_size)
            ]
            futures = [executor.submit(self._embed_batch, batch) for batch in batches]
            results = [future.result() for future in futures]
        return [emb for batch in results for emb in batch]

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts (used by ThreadPoolExecutor).

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors as lists of floats
        """
        # Can use self.model directly since we're using threads
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]


@timeit
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


@timeit
def extract_embed_n_save(pdf_path, chunk_size, model_name, pdf_filename):
    # Store raw text
    print("extract_embed_n_save Started")
    extracted_text = extract_text_from_pdf(pdf_path)
    fname_without_ext = os.path.splitext(pdf_filename)[0]
    text_filename = fname_without_ext + ".txt"
    store_text(extracted_text, text_filename)
    start_time = time.time()
    # Clean text using enhanced cleaner and store cleaned version
    cleaned_text = TextCleaner.clean_text_aggressive(extracted_text)
    print(f"TT 4 aggressive cleaning : {time.time() - start_time} seconds")

    cleaned_text_filename = fname_without_ext + ".txt"
    store_cleaned_text(cleaned_text, cleaned_text_filename)

    # Advanced chunking on cleaned text
    chunks = advanced_chunk_text(cleaned_text, chunk_size=chunk_size)

    # Embed and store vectors using cleaned chunks
    embedder = get_embedder(model_name)
    vectors = embedder.embed_async(chunks)
    vector_filename = fname_without_ext + ".npy"
    store_vectors(vectors, vector_filename)

    print(
        f"Saved raw text to {text_filename}, cleaned text to {cleaned_text_filename}, and vectors to {vector_filename}"
    )
    print(f"Generated {len(chunks)} cleaned chunks from {len(chunks)} original chunks")

    if SAVE_TO_DB:
        save_in_db(pdf_filename, chunk_size, chunks, vectors)


@timeit
def save_in_db(pdf_filename, chunk_size, chunks, vectors):
    print(f"Saving {pdf_filename} to db")
    org_id = get_or_create_org("default")
    pdf_id = insert_pdf(org_id, pdf_filename, chunk_size)

    # Prepare data for bulk insert
    chunks_data = []
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        # Final safety check - clean the chunk one more time before saving
        if chunk.strip():  # Only save non-empty chunks
            chunks_data.append((idx, chunk, vector))

    # Bulk insert all chunks with embeddings in one operation
    start_time = time.time()
    success = bulk_insert_chunks_with_embeddings_copy(pdf_id, chunks_data)
    end_time = time.time()

    if success:
        print(
            f"Bulk inserted {len(chunks_data)} chunks in {end_time - start_time:.2f} seconds"
        )
        print(f"Saved {pdf_filename} to db with {len(chunks_data)} cleaned chunks")
    else:
        print(f"Error: Failed to bulk insert chunks for {pdf_filename}")


def cosine_sim(a: List[float], b: List[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


@timeit
def get_top_k_indices(scores: List[float], k: int) -> List[int]:
    arr = np.array(scores)
    return arr.argsort()[-k:][::-1].tolist()


# Usage:
# embedder = Embedder()
# vectors = embedder.embed(["chunk1", "chunk2"])
