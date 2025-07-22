import numpy as np
from typing import List

def cosine_sim(a: List[float], b: List[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def get_top_k_indices(scores: List[float], k: int) -> List[int]:
    arr = np.array(scores)
    return arr.argsort()[-k:][::-1].tolist() 