# QueryDoc

This project is an example chatbot that analyzes PDF documents using an embedding model and generates LLM answers through a Coarse-to-Fine search (RAG) approach.

## QueryDoc
```bash
QueryDoc/
├─ scripts/
│   ├─ pdf_extractor.py
│   ├─ chunker.py
│   ├─ build_index.py
│   └─ section_rep_builder.py
├─ src/
│   ├─ inference/
│   │   ├─ embedding_model.py
│   │   └─ llm_model.py
│   ├─ search/
│   │   ├─ section_coarse_search.py
│   │   ├─ fine_search.py
│   │   └─ vector_search.py
│   ├─ chatbot.py
│   └─ utils/
│       ├─ init.py
│       └─ text_cleaning.py
├─ data/
│   ├─ extracted/
│   ├─ chunks/
│   ├─ index/
│   └─ original/
│       └─ sample.pdf
├─ app.py
├─ requirements.txt
└─ README.md
```

## Installation and Execution

1. Create a virtual environment and install libraries
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
• For OCR features, install Tesseract and the appropriate language data, e.g.  
  `sudo apt-get install tesseract-ocr`   # Debian/Ubuntu
  `sudo apt-get install tesseract-ocr-kor`
  `brew install tesseract-ocr`   # MacOS
  `brew install tesseract-ocr-kor` 
(On Windows, activate with .\venv\Scripts\activate or a similar command.)

2.	Extract PDF & Split into Chunks
```bash
python scripts/pdf_extractor.py
python scripts/chunker.py
```
•	After execution, JSON files will be created in data/extracted/*.json and data/chunks/*.json.

3.	Build Embeddings
```bash
python scripts/build_index.py
```
•	This generates data/index/*_vectors.json.

4.	Generate Section Representative Vectors
```bash
python scripts/section_rep_builder.py
```
•	This creates files like sections_with_emb.json.


5. Directly Testing chatbot.py

To test the chatbot interactively, run:
```bash
python src/chatbot.py
```

•	Then type your question (e.g., “Explain Chapter 1.”) and press Enter. 

•	Type “exit” to quit.

The system prompt used by the chatbot can be customized by editing `DEFAULT_SYSTEM_PROMPT` in `src/chatbot.py` or by passing a custom prompt when creating a `PDFChatBot` instance.

6.	Run the Chatbot Server
```bash
python app.py
```
    
• A FastAPI server will run at http://0.0.0.0:8000 (default port).

• You can send a JSON-formatted question to the POST /ask endpoint to receive an answer.

7. Launch the Web Demo
```bash
python web_demo.py
```

• A browser window will appear allowing you to upload a PDF and edit the system prompt before asking questions.

8. Example API Request
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What's egocentric AI agent?"}'
```

9. Additional Web Demo Info
```bash
python web_demo.py
```
* Sign in with the default credentials `admin`/`password`.
* Uploaded PDFs are saved under `data/user_uploads/<username>`.
* You can modify the system prompt in `src/chatbot.py` or provide one in the web interface.

## Key Libraries

• PyMuPDF (fitz): Extracts PDF text and table of contents (ToC).

• SentenceTransformers: Loads the BAAI/bge-m3 embedding model.

• Transformers: Provides the local LLM (e.g., trillionlabs/Trillion-7B-preview).

• FastAPI: A simple REST API server.

• Gradio: Interactive web demo framework.

• pdfplumber: Layout‑aware PDF parsing  
• pytesseract: OCR fallback engine  
• Pillow: Image handling for OCR pipelines  
• pandas: DataFrame operations for layout analysis  
• scikit‑learn: KMeans clustering for multi‑column detection

## Notes

• Models such as bge-m3 and Trillion-7B may take some time to download the first time they are loaded.

• Since section content is complemented using the average of section chunk embeddings (without a summarization model), very long sections may result in reduced search accuracy. (Utilizing a summarization model may be considered in the future.)
