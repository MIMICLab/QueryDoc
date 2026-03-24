# src/search/vector_search.py

import numpy as np
from typing import List, Dict

def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot = np.dot(v1, v2)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
    return dot / denom

def simple_vector_search(query_emb, index_data: List[Dict], top_k=8):
    """
    query_emb: numpy array or list[float]
    index_data: [{"embedding": [...], "metadata": {...}}, ...]
    """
    results = []
    query_vec = np.array(query_emb)
    q_norm = np.linalg.norm(query_vec)

    for item in index_data:
        emb = np.array(item["embedding"])
        dot = np.dot(query_vec, emb)
        denom = np.linalg.norm(emb) * q_norm + 1e-8
        score = dot / denom
        results.append((score, item))

    results.sort(key=lambda x: x[0], reverse=True)
    top_results = [r[1] for r in results[:top_k]]
    return top_results