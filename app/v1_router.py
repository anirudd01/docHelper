from fastapi import APIRouter, Query
from utils.vector_utils import get_embedder
from utils.file_manager import list_pdfs, read_cleaned_chunks_and_vectors
from utils.llm_utils import generate_llm_answer
from utils.vector_utils import cosine_sim, get_top_k_indices

v1_router = APIRouter(prefix="/v1")


@v1_router.post("/ask")
def ask_question(
    question: str = Query(..., description="Your question about the PDF"),
    top_k: int = Query(3, description="Number of top chunks to use as context"),
    chunk_size: int = Query(200, description="Words per chunk (advanced chunking)"),
    overlap: int = Query(
        30, description="Words to overlap between chunks (advanced chunking)"
    ),
):
    pdfs = list_pdfs()
    if not pdfs:
        return {"error": "No PDFs available."}
    q_vec = get_embedder().embed([question])[0]
    all_chunks = []
    all_embeddings = []
    chunk_sources = []
    for pdf in pdfs:
        result = read_cleaned_chunks_and_vectors(
            pdf, chunk_size=chunk_size, overlap=overlap
        )
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
        answer = generate_llm_answer(prompt)
        return {
            "answer": answer,
            # "context_chunks": [all_chunks[i] for i in top_indices],
            "sources": list(set([chunk_sources[i] for i in top_indices])),
        }
    except Exception as e:
        return {"error": str(e)}
