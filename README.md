# 🧠 docHelper - Local PDF Q&A App using Ollama + FastAPI

A **learning-first project** to explore integrating **Large Language Models (LLMs)** into Python applications. This app enables you to upload PDFs, extract their content, and ask questions about them using a **locally running LLM (via [Ollama](https://ollama.com))** — no internet APIs or external databases required (at first).

---

## 🚦 Current State & Features

### ✅ Accomplished (v1 + v2 + v3 + v7 + parts of v4)
- Upload and manage multiple PDFs in file system (no overwrite, filenames preserved)
- Extract and store text from PDFs for fast, repeatable access
- Chunk text at upload time (user-defined chunk size, fixed per file)
- Generate and store vector embeddings for each PDF (background task)
- Efficient vector search for Q&A (RAG) across selected or all PDFs
- Ask questions using local LLM (Ollama) with relevant context
- Modular code: utilities for file management, embedding, vector search, and LLM calls
- All file I/O and chunk/vector management handled in utils (not main API logic)
- **✅ v3: Database-Backed RAG with PostgreSQL + pgvector for persistent, scalable vector search**
- **✅ v7: Model Deployment & Cloud Readiness - Deployed on Railway with production-ready setup**
- **✅ Parts of v4: Text Cleaning & Intelligent Chunking - Basic text preprocessing implemented**
- **✅ Streamlit UI: Modern web interface for PDF upload and Q&A interactions**
- **✅ Railway Deployment: Both frontend (Streamlit) and backend (FastAPI) deployed and tested**
- **✅ Performance Optimizations: PyMuPDF for fast PDF extraction (1.04s vs 18s), COPY operations for bulk database inserts (2.24s vs 65s), async embedding with ThreadPoolExecutor**
- **✅ Hybrid PDF Processing: PyMuPDF for speed with pdfplumber fallback for accuracy**

---

## 🚀 Performance Optimizations Achieved

### 📊 Performance Improvements
- **PDF Extraction**: 18.52s → 1.04s (**94% faster**) using PyMuPDF with streaming
- **Database Operations**: 65s → 2.24s (**97% faster**) using PostgreSQL COPY commands
- **Embedding Generation**: 118s → 5.24s (**96% faster**) using async processing with ThreadPoolExecutor
- **Total Processing Time**: 213s → ~15-20s (**90%+ faster**) for large documents

### 🔧 Technical Optimizations
- **Hybrid PDF Processing**: PyMuPDF for speed, pdfplumber fallback for accuracy
- **Streaming PDF Extraction**: Load pages individually instead of entire PDF
- **Bulk Database Operations**: COPY commands instead of individual INSERTs
- **Async Embedding**: Parallel processing with ThreadPoolExecutor
- **Smart Batching**: Automatic batch size optimization based on CPU cores

### 📈 Scalability Improvements
- **Memory Efficiency**: Streaming processing reduces memory usage
- **Database Performance**: Single connection with bulk operations
- **Concurrent Processing**: Parallel embedding generation
- **Error Handling**: Robust fallbacks for PDF extraction

---

## 🛣️ Roadmap & Next Phases

### 🧹 v4: Enhanced Text Cleaning & Intelligent Chunking (In Progress)
- ✅ Basic text cleaning implemented (noise removal, formatting fixes, bullet point removal)
- 🔄 Support advanced chunking strategies (semantic, section-aware)
- 🔄 Store cleaned text for future use
- 🔄 Improves RAG and LLM answer quality

### 🧠 v5: Prompt Engineering & Domain Guardrails
- Add prompt engineering techniques to ensure only company/employee domain questions are answered
- Out-of-domain questions receive a polite, safe response
- Add prompt templates and context filtering

### 🚀 v6: Large File & Performance Testing
- Test with large PDFs (300-500+ pages)
- Optimize chunking, embedding, and vector search for speed and memory
- Monitor and profile for bottlenecks

### 💻 v8: Enhanced UI, Auth, and User Experience
- ✅ Basic Streamlit UI implemented
- 🔄 Implement authentication and per-user chat history
- 🔄 Remember chat and context for each user
- 🔄 Add admin features for managing orgs, files, and users

---

## 🚀 Deployment & Usage

### Local Development
```bash
# Backend (FastAPI)
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend (Streamlit)
cd streamlit_ui
pip install -r requirements.txt
streamlit run streamlit_app.py --server.port 8123 --server.address 0.0.0.0
```

### Production Deployment (Railway)
- **Backend**: FastAPI app deployed on Railway with PostgreSQL + pgvector
- **Frontend**: Streamlit app deployed on Railway with environment variables for API communication
- **Database**: PostgreSQL with pgvector extension for vector similarity search
- **Environment**: Production-ready with proper environment variable configuration

### API Endpoints
- `POST /core/upload-pdf` - Upload PDF files
- `GET /core/list-pdfs` - List uploaded PDFs
- `GET /core/get-pdf-text/{filename}` - Get extracted text from PDF
- `GET /core/download-pdf/{filename}` - Download PDF file
- `DELETE /core/remove-pdf` - Remove PDF and associated data (text, vectors, DB records)
- `POST /core/clean-chunks` - Clean existing chunks in batch of 10 and regenerate vectors in database
- `POST /v1/ask` - File-based Q&A (legacy)
- `POST /v2/ask_ai` - Database-backed Q&A with vector search

---

## 🧰 Best Practices & Learning Goals
- Keep file and vector management modular and in utils
- Store all metadata (including chunk size) for reproducibility
- Use background tasks for heavy processing (embedding, text extraction)
- Separate API logic from file/data logic
- Always preprocess and sanitize text before chunking/embedding
- Use database for metadata and vector search at scale
- Test with large files and real-world data
- Add guardrails and prompt engineering for safe, relevant answers
- Build incrementally, with clear phases and goals

---

Happy building and learning! 🚀

---

## 📁 File/Folder Overview

- `app/main.py` — FastAPI app entrypoint, only mounts routers, no endpoint logic except health check.
- `app/core_router.py` — Core endpoints: upload, list, get-pdf-text (prefix: /core)
- `app/v1_router.py` — v1 endpoints: file-based /ask (prefix: /v1)
- `app/v2_router.py` — v2 endpoints: DB/pgvector based /ask_ai (prefix: /v2)
- `utils/file_manager.py` — Handles PDF/text/vector storage, chunking, and preview utilities.
- `utils/vector_utils.py` — Cosine similarity and top-k selection utilities for vector search and Embedding logic, background processing, and DB/file save logic.
- `utils/llm_utils.py` — Handles LLM (Ollama) prompt calls and streaming output.
- `utils/db_utils.py` — PostgreSQL/pgvector connection, table creation, and DB insert/query helpers.
- `utils/db_models.py` — Python dataclasses for Org, PDF, Chunk, Embedding (for structuring data in code).
- `utils/text_cleaner.py` — Text preprocessing utilities for cleaning and formatting PDF text.
- `streamlit_ui/streamlit_app.py` — Streamlit web interface for PDF upload and Q&A interactions.
- `streamlit_ui/requirements.txt` — Python dependencies for the Streamlit frontend.
- `data/pdfs/` — Directory where all uploaded PDFs are stored.
- `data/texts/` — Directory for extracted text files.
- `data/vectors/` — Directory for vector files (.npy).
- `requirements.txt` — Python dependencies for the FastAPI backend.
- `README.md` — Project documentation, roadmap, and usage instructions.

### How to run Streamlit app locally
```bash
cd streamlit_ui
streamlit run streamlit_app.py --server.showEmailPrompt False --server.enableXsrfProtection=false --server.enableCORS=false --server.port 8123 --server.address 0.0.0.0
```
