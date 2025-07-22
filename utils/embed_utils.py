from sentence_transformers import SentenceTransformer
from typing import List, Optional

# Default embedding model and size (can be changed later)
DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

class Embedder:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or DEFAULT_EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

# Usage:
# embedder = Embedder()
# vectors = embedder.embed(["chunk1", "chunk2"]) 