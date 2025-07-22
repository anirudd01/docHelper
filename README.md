# 🧠 docHelper - Local PDF Q&A App using Ollama + FastAPI (Python 3.13)

A **learning-first project** to explore integrating **Large Language Models (LLMs)** into Python applications. This app enables you to upload a PDF, extract its content, and ask questions about it using a **locally running LLM (via [Ollama](https://ollama.com))** — no internet APIs or external databases required (at first).

---

## 📚 Project Overview

This project is structured in **three progressive phases/versions** to help you learn and build incrementally:

### 🚦 Version 1 (v1): Single PDF Only
- The app works with just one PDF at a time.
- All questions are answered using the content or vectors of that single PDF.
- Uploading a new PDF replaces the previous one.

### 📂 Version 2 (v2): Multiple PDFs, User-Selectable
- The app supports multiple uploaded PDFs.
- The user can select which PDF(s) to query for answers.
- Questions are answered using the selected PDF(s) only.

### 🧠 Version 3 (v3): Database-Backed, Global Search
- The app uses a database (PostgreSQL + pgvector) to store all PDF contents and vectors.
- The user cannot select specific PDFs; instead, all questions are answered using the most relevant data from any available PDF.
- This enables global, context-aware search across all uploaded documents.

---

## 🧰 Tech Stack

| Purpose              | Tool/Library                                   |
|----------------------|------------------------------------------------|
| Web API              | [FastAPI](https://fastapi.tiangolo.com/)       |
| Local LLM API        | [Ollama](https://ollama.com)                   |
| PDF Text Extraction  | `pdfplumber` or `PyPDF2`                       |
| Embeddings           | `sentence-transformers`                        |
| Vector Search        | FAISS (in-memory), pgvector (later)            |
| Optional UI          | Swagger UI, or basic frontend (future)         |

---

## ✅ Prerequisites

- Python 3.13
- [Ollama installed and running](https://ollama.com)
- At least one model pulled locally (e.g., `llama3`, `mistral`, or `gemma`)

```bash
ollama run llama3  # or mistral, gemma, etc.
```

---

## 📁 Suggested Folder Structure

```text
askmypdf/
├── app/
│   ├── main.py             # FastAPI routes
│   ├── pdf_utils.py        # PDF reading & text chunking
│   ├── embed_utils.py      # Text embedding logic
│   ├── llm_utils.py        # Prompting and interaction with Ollama
│   └── vector_store.py     # In-memory or FAISS search
├── data/
│   └── pdfs/      # PDF files saved here
├── requirements.txt
└── README.md
```

---

## 🚦 Phase 1: PDF Upload & Text Extraction

### Goals
- **v1:** Upload a single PDF via API (replaces previous PDF)
- Extract raw text
- Chunk the text (if large)
- Store text temporarily (in memory or file)

### Implementation Steps
1. **Initialize FastAPI app**
2. **Create `/upload-pdf` endpoint** (single PDF in v1)
3. **Extract text** using `pdfplumber` or `PyPDF2`
4. **Chunk text** (e.g., by 500 tokens or ~200 words)
5. **Store chunks** in memory (global dict keyed by session/filename)
6. **Save original PDF** in `/data/uploaded_pdfs/`

---

## 🔍 Phase 2: Local RAG with Ollama (No DB)

### Goals
- **v2:** Support multiple PDFs, allow user to select which PDF(s) to query
- Accept a question
- Embed the question and compare with PDF chunks
- Use Ollama to answer using most relevant chunks

### Implementation Steps
1. **Generate embeddings** for each text chunk using `sentence-transformers`
2. **Use cosine similarity** to find top-k relevant chunks for a question
3. **Build `/ask` endpoint`** (add PDF selection in v2)
4. **Create a prompt template:**

   ```text
   Context:
   <top relevant chunk(s)>

   Question: <user question>
   Answer:
   ```
5. **Send prompt to Ollama** using its HTTP API
6. **Return the LLM-generated answer**

---

## 💾 Phase 3: Persist with pgvector + PostgreSQL

### Goals
- **v3:** User cannot select PDFs; queries search all available PDFs for relevant data
- Replace in-memory vector search with pgvector
- Enable multiple PDFs, persistent storage

### Implementation Steps
1. **Install PostgreSQL** with pgvector extension
2. **Create tables:** PDFs, Chunks, Embeddings
3. **Store chunk embeddings in DB**
4. **Use vector similarity SQL** to retrieve context
5. **Add metadata** (page number, filename, upload time, etc.)

---

## 📦 Installation

### 1. Clone this Repo

```bash
git clone https://github.com/your-username/askmypdf.git
cd askmypdf
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Sample `requirements.txt`:**

```
fastapi
uvicorn
pdfplumber
sentence-transformers
scikit-learn
requests
```

### 4. Run the App

```bash
uvicorn app.main:app --reload
```

Swagger UI will be available at [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🤖 Connecting to Ollama

Ollama runs locally at `http://localhost:11434`.

**Example API call:**

```python
import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3",
        "prompt": "Context: ... \n\nQuestion: ... \nAnswer:"
    }
)
print(response.json()['response'])
```

---

## 🧪 Sample Usage

### v1: Single PDF
1. Upload a PDF via `/upload-pdf` (this replaces any previous PDF)
2. Ask a question (all questions use the current PDF)

### v2: Multiple PDFs
1. Upload multiple PDFs
2. Select which PDF(s) to use for answering questions
3. Ask a question (answers use only the selected PDFs)

### v3: Database-Backed, Global Search
1. Upload PDFs (all are stored in the database)
2. Ask a question (the app finds the most relevant context from any PDF, user cannot select PDFs)

---

## ✨ Stretch Goals (Optional)

- [ ] Add session or file-specific context tracking
- [ ] Use LangChain or LlamaIndex to simplify the pipeline
- [ ] Use Docker for local Ollama + app setup
- [ ] Add a basic frontend (Streamlit or HTML form)
- [ ] Replace Ollama with OpenAI/Gemini API

---

## 🙌 Contributing

This is a personal learning project. Feedback and suggestions are welcome! Feel free to open issues or submit pull requests.

---

## 🧠 TL;DR

- **v1:** Upload PDFs → Only one PDF active at a time, all questions use that PDF
- **v2:** Multiple PDFs, user can select which PDFs to use for answers
- **v3:** All PDFs stored in DB, user cannot select PDFs, answers use most relevant context from any PDF
- Use sentence embeddings for similarity
- Use Ollama locally to answer questions with relevant context
- Start simple, scale with pgvector later

---

Happy building! 🚀

---

*Let me know when you're ready and I’ll help you scaffold `main.py`, `pdf_utils.py`, and others in **Phase 1** — with minimal magic and full explainability.*
