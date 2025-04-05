# src/search/section_coarse_search.py

import numpy as np
from src.inference.kure_embedding_model import kure_embedding_model

def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot = np.dot(v1, v2)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
    return dot / denom

def coarse_search_sections(query: str, sections: list, beta=0.3, top_k=3):
    """
    sections: [
      { "title": "...", "title_emb": [...], "avg_chunk_emb": [...], ... },
      ...
    ]
    query 임베딩과 섹션(title_emb, avg_chunk_emb) 간 코사인 유사도를 각각 구해
    final_score = beta * sim_title + (1 - beta) * sim_chunk
    상위 top_k 섹션 반환
    """
    query_emb = kure_embedding_model.get_embedding(query)

    scored = []
    for sec in sections:
        title_emb = sec.get("title_emb")
        chunk_emb = sec.get("avg_chunk_emb")
        if title_emb is None or chunk_emb is None:
            # 데이터가 없는 경우는 패스
            continue
        sim_title = cosine_similarity(query_emb, title_emb)
        sim_chunk = cosine_similarity(query_emb, chunk_emb)

        final_score = beta * sim_title + (1 - beta) * sim_chunk
        scored.append((final_score, sec))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    top_sections = [x[1] for x in scored[:top_k]]
    return top_sections