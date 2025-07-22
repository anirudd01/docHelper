from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, FastAPI, File, Query, UploadFile

from utils.embed_utils import (
    DEFAULT_EMBEDDING_MODEL,
    extract_embed_n_save,
    get_embedder,
)
from utils.file_manager import (
    extract_text_from_pdf,
    fetch_pdf,
    list_pdfs,
    preview_lines,
    read_chunks_and_vectors,
    store_pdf,
)
from utils.llm_utils import generate_llm_answer
from utils.vector_utils import cosine_sim, get_top_k_indices

app = FastAPI()
router = APIRouter(prefix="/v1")

# Global embedder object (default model)
EMBEDDING_MODEL = str(DEFAULT_EMBEDDING_MODEL)
EMBEDDER = None


@router.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    chunk_size: int = Query(200, description="Words per chunk"),
    embedding_model: Optional[str] = Query(
        EMBEDDING_MODEL, description="SentenceTransformer model name"
    ),
    background_tasks: BackgroundTasks = None,
):
    file_bytes = await file.read()
    pdf_path = store_pdf(file_bytes, file.filename)
    # Background task: store text and vectors
    background_tasks.add_task(
        extract_embed_n_save, pdf_path, chunk_size, embedding_model, file.filename
    )
    return {
        "filename": file.filename,
        "status": "uploaded",
        "embedding_model": embedding_model,
        "embedding_in_background": True,
    }


@router.get("/list-pdfs", response_model=List[str])
def list_uploaded_pdfs():
    return list_pdfs()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/get-pdf-text")
def get_pdf_text(filename: str = Query(...)):
    pdf_path = fetch_pdf(filename)
    if not pdf_path:
        return {"error": "File not found"}
    text = extract_text_from_pdf(pdf_path)
    preview = preview_lines(text, n=5)
    return {
        "filename": filename,
        "preview": preview,
        "total_lines": len(text.splitlines()),
    }


@router.post("/ask")
def ask_question(
    question: str = Query(..., description="Your question about the PDF"),
    top_k: int = Query(3, description="Number of top chunks to use as context"),
):
    pdfs = list_pdfs()
    if not pdfs:
        return {"error": "No PDFs available."}
    q_vec = get_embedder().embed([question])[0]
    all_chunks = []
    all_embeddings = []
    chunk_sources = []
    for pdf in pdfs:
        result = read_chunks_and_vectors(pdf)
        if not result:
            print(f"WARNING!!! no chunks or vectors found {pdf}")
            continue
        chunks, vectors = result
        all_chunks.extend(chunks)
        all_embeddings.extend(vectors)
        chunk_sources.extend([pdf] * len(chunks))
    if not all_chunks or not all_embeddings:
        return {"error": "No chunks or embeddings found for selected PDFs."}
    sims = [cosine_sim(q_vec, emb) for emb in all_embeddings]
    top_indices = get_top_k_indices(sims, top_k)
    context = "\n\n".join( 
        [f"[{chunk_sources[i]}] {all_chunks[i]}" for i in top_indices]
    )
    prompt = f"""Context:\n{context}\n\nQuestion: {question}\nAnswer:"""
    try:
        answer = generate_llm_answer(prompt, model_name="llama3.2")
        return {
            "answer": answer,
            "context_chunks": [all_chunks[i] for i in top_indices],
            "sources": [chunk_sources[i] for i in top_indices],
        }
    except Exception as e:
        return {"error": str(e)}


app.include_router(router)
get_embedder()
print("Loaded embedder")
