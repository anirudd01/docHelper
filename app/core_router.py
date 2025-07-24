from fastapi import APIRouter, BackgroundTasks, File, Query, UploadFile
from fastapi.responses import FileResponse
from typing import List, Optional
from utils.vector_utils import DEFAULT_EMBEDDING_MODEL, extract_embed_n_save
from utils.file_manager import (
    extract_text_from_pdf,
    fetch_pdf,
    list_pdfs,
    preview_lines,
    store_pdf,
)

core_router = APIRouter(prefix="/core")


@core_router.get("/list-pdfs", response_model=List[str])
def list_uploaded_pdfs():
    return list_pdfs()


@core_router.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    chunk_size: int = Query(200, description="Words per chunk"),
    embedding_model: Optional[str] = Query(
        DEFAULT_EMBEDDING_MODEL, description="SentenceTransformer model name"
    ),
    background_tasks: BackgroundTasks = None,
):
    file_bytes = await file.read()
    pdf_path = store_pdf(file_bytes, file.filename)
    background_tasks.add_task(
        extract_embed_n_save, pdf_path, chunk_size, embedding_model, file.filename
    )
    return {
        "filename": file.filename,
        "status": "uploaded",
        "embedding_model": embedding_model,
        "embedding_in_background": True,
    }


@core_router.get("/get-pdf-text")
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


@core_router.get("/download-pdf")
def download_pdf(filename: str = Query(...)):
    pdf_path = fetch_pdf(filename)
    if not pdf_path:
        return {"error": "File not found"}
    return FileResponse(pdf_path, media_type="application/pdf", filename=filename)
