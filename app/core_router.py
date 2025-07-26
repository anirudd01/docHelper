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
    remove_pdf_files,
)
from utils.db_utils import remove_pdf_data, get_or_create_org

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


@core_router.delete("/remove-pdf")
def remove_pdf(
    filename: str = Query(..., description="Name of the PDF file to remove"),
    org_name: str = Query("default", description="Organization name"),
):
    """
    Remove a PDF file and its associated data.
    
    This endpoint will:
    1. Remove text files, cleaned text files, and vector files from the file system
    2. Mark the PDF as inactive in the database
    3. Remove chunks and embeddings from the database
    
    Returns:
        dict: Status of the operation with details about what was removed
    """
    # Check if PDF exists in file system
    pdf_path = fetch_pdf(filename)
    if not pdf_path:
        return {"error": "PDF file not found in storage"}
    
    # Get org ID
    org_id = get_or_create_org(org_name)
    
    # Remove database records
    db_result = remove_pdf_data(filename, org_id)
    
    # Remove file system files
    file_result = remove_pdf_files(filename)
    
    # Combine results
    if db_result["success"] and file_result["success"]:
        return {
            "success": True,
            "message": f"Successfully removed PDF '{filename}' and all associated data",
            "database": db_result,
            "filesystem": file_result,
        }
    else:
        return {
            "success": False,
            "message": "Some operations failed",
            "database": db_result,
            "filesystem": file_result,
        }
