# scripts/chunker.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from typing import List, Dict, Any
from src.utils.text_cleaning import basic_clean_text

CHUNK_SIZE = 500  # 글자 기준 (예: 500자씩)

def get_section_of_page(page_num: int, toc: List[List[Any]]) -> str:
    """
    PDF ToC: [(level, title, start_page), ...]
    page_num는 0-based, start_page는 1-based
    page_num+1 >= start_page => 해당 섹션
    단순한 방식으로 현재 섹션 제목을 식별
    """
    current_section = "기타"
    for (lvl, title, start_p) in toc:
        if page_num + 1 >= start_p:
            current_section = title
        else:
            break
    return current_section

def chunk_text(text: str, chunk_size: int) -> List[str]:
    text = basic_clean_text(text)
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_str = text[start:end].strip()
        if chunk_str:
            chunks.append(chunk_str)
        start = end
    return chunks

def process_extracted_file(json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    json_data: {
      "file_path": "...",
      "toc": [(level, title, start_page), ...],
      "pages_text": ["page0 text", "page1 text", ...]
    }
    """
    pdf_path = json_data["file_path"]
    toc = json_data["toc"]
    pages_text = json_data["pages_text"]

    chunked_result = []
    for page_idx, text in enumerate(pages_text):
        section_title = get_section_of_page(page_idx, toc)
        # chunkify
        splitted = chunk_text(text, CHUNK_SIZE)
        for c_i, c_text in enumerate(splitted):
            chunked_result.append({
                "file_path": pdf_path,
                "page_idx": page_idx,
                "section_title": section_title,
                "chunk_index": c_i,
                "content": c_text
            })
    return chunked_result

if __name__ == "__main__":
    extracted_folder = "data/extracted"
    chunk_folder = "data/chunks"
    os.makedirs(chunk_folder, exist_ok=True)

    for fname in os.listdir(extracted_folder):
        # sections.json 파일은 건너뛰기
        if fname.endswith(".json") and fname != "sections.json":
            path = os.path.join(extracted_folder, fname)
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            chunked_data = process_extracted_file(data)

            base_name = os.path.splitext(fname)[0]
            out_json = os.path.join(chunk_folder, f"{base_name}_chunks.json")
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(chunked_data, f, ensure_ascii=False, indent=2)

    print("Chunking Complete.")