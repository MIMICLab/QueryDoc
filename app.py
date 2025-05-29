# app.py

import uvicorn
import json
import os
from fastapi import FastAPI, Body

from scripts import section_rep_builder
from src.chatbot import PDFChatBot

app = FastAPI()

# 1) Load section information (sections_with_emb.json)
sections_path = "data/extracted/sections_with_emb.json"
with open(sections_path, 'r', encoding='utf-8') as f:
    sections_data = json.load(f)

# 2) Load chunk index (sample_chunks_vectors.json)
# section_rep_builder 매서드를 가져옴 / 파일 이름이 바뀌어도 실행 가능
chunk_index_path = section_rep_builder.find_one_vectors_file(index_dir="data/index")
with open(chunk_index_path, 'r', encoding='utf-8') as f:
    chunk_index_data = json.load(f)

chatbot = PDFChatBot(sections_data, chunk_index_data)

@app.post("/ask")
def ask_question(question: str = Body(..., embed=True)):
    """
    FastAPI endpoint that returns an answer for the given question.

    Parameters
    ----------
    question : str
        The user’s question text, passed in the request body.

    Returns
    -------
    dict
        A JSON dictionary with a single key ``"answer"``.
    """
    answer = chatbot.answer(question)
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)