# 🧠 docHelper - Local PDF Q&A App using Ollama + FastAPI

A **learning-first project** to explore integrating **Large Language Models (LLMs)** into Python applications. This app enables you to upload PDFs, extract their content, and ask questions about them using a **locally running LLM (via [Ollama](https://ollama.com))** — no internet APIs or external databases required (at first).

---

## 🚦 Current State & Features

### ✅ Accomplished (v1 + v2)
- Upload and manage multiple PDFs in file system (no overwrite, filenames preserved)
- Extract and store text from PDFs for fast, repeatable access
- Chunk text at upload time (user-defined chunk size, fixed per file)
- Generate and store vector embeddings for each PDF (background task)
- Efficient vector search for Q&A (RAG) across selected or all PDFs
- Ask questions using local LLM (Ollama) with relevant context
- Modular code: utilities for file management, embedding, vector search, and LLM calls
- All file I/O and chunk/vector management handled in utils (not main API logic)
- Ready for database integration and advanced features

---

## 🛣️ Roadmap & Next Phases

### 🚦 v3: Database-Backed RAG with ORG Support
- Integrate PostgreSQL + pgvector for persistent, scalable vector search
- Add org info (start with one org, scalable to many)
- Store files on disk, but keep extensive metadata in DB (filename, org, chunk size, upload time, etc.)
- Separate table for vector search (chunk, embedding, metadata)
- /ask endpoint remains as is (file-based, for backward compatibility)
- New /ask_ai endpoint: retrieves context from DB, then queries LLM
- Best practices: use DB for metadata, keep files on disk for large file support, always track chunk size
- Prepare for scaling to 100s/1000s of PDFs and multi-org use

### 🧹 v4: Text Cleaning & Intelligent Chunking
- Add text_cleaner utility for preprocessing (remove noise, fix formatting, categorize sections)
- Support advanced chunking strategies (semantic, section-aware)
- Store cleaned text for future use
- Improves RAG and LLM answer quality

### 🧠 v5: Prompt Engineering & Domain Guardrails
- Add prompt engineering techniques to ensure only company/employee domain questions are answered
- Out-of-domain questions receive a polite, safe response
- Add prompt templates and context filtering

### 🚀 v6: Large File & Performance Testing
- Test with large PDFs (300-500+ pages)
- Optimize chunking, embedding, and vector search for speed and memory
- Monitor and profile for bottlenecks

### 🛡️ v7: Model Deployment & Cloud Readiness
- Explore loading models directly (not via Ollama API)
- Prepare for deployment on platforms like Railway (with/without Docker)
- If Ollama is not deployable, support model files and custom inference interface
- Add deployment scripts and best practices

### 💻 v8: UI, Auth, and User Experience
- Add a simple UI (Streamlit or web frontend)
- Implement authentication and per-user chat history
- Remember chat and context for each user
- Add admin features for managing orgs, files, and users

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
- `data/pdfs/` — Directory where all uploaded PDFs are stored.
- `data/texts/` — Directory for extracted text files.
- `data/vectors/` — Directory for vector files (.npy).
- `requirements.txt` — Python dependencies for the project.
- `README.md` — Project documentation, roadmap, and usage instructions.
