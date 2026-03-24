# scripts/section_rep_builder.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import numpy as np
from src.inference.embedding_model import embedding_model

def build_section_reps(sections, chunk_index):
    """
    sections: [
      { "title": "2장 설치방법", "start_page":10, "end_page":19, ... },
      ...
    ]
    chunk_index: [{ "embedding": [...], "metadata": {"section_title": "...", ...}}, ...]

    => 각 섹션에
       sec["title_emb"], sec["avg_chunk_emb"] 필드를 추가해 반환
    """
    # 1) 섹션 제목 임베딩 (batch)
    titles = [sec["title"] for sec in sections]
    title_embs = embedding_model.get_embeddings(titles)  # shape: (num_sections, dim)
    for i, sec in enumerate(sections):
        sec["title_emb"] = title_embs[i].tolist()

    # 2) 섹션별 청크 모으기
    section2embs = {}
    for item in chunk_index:
        sec_t = item["metadata"]["section_title"]
        emb = item["embedding"]  # list[float]
        if sec_t not in section2embs:
            section2embs[sec_t] = []
        section2embs[sec_t].append(emb)

    # 3) 섹션 내부 청크들의 평균 임베딩
    for sec in sections:
        stitle = sec["title"]
        if stitle not in section2embs:
            sec["avg_chunk_emb"] = None
        else:
            arr = np.array(section2embs[stitle])  # shape: (num_chunks, emb_dim)
            avg_vec = arr.mean(axis=0)            # (emb_dim,)
            sec["avg_chunk_emb"] = avg_vec.tolist()
    
    return sections

if __name__ == "__main__":
    # 예시: data/extracted/sections.json (목차 기반 섹션 정보)
    sections_json = "data/extracted/sections.json"
    # 예시: data/index/sample_chunks_vectors.json (청크 임베딩)
    chunk_index_json = "data/index/sample_chunks_vectors.json"

    with open(sections_json, 'r', encoding='utf-8') as f:
        sections_data = json.load(f)
    
    with open(chunk_index_json, 'r', encoding='utf-8') as f:
        chunk_index_data = json.load(f)

    # 섹션 대표 벡터 생성
    updated_sections = build_section_reps(sections_data, chunk_index_data)

    # 저장(예: data/extracted/sections_with_emb.json)
    out_path = "data/extracted/sections_with_emb.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(updated_sections, f, ensure_ascii=False, indent=2)

    print("Section reps built and saved.")