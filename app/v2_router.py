from fastapi import APIRouter, Query
from utils.vector_utils import get_embedder
from utils.db_utils import get_db_conn, get_or_create_org
from utils.llm_utils import generate_llm_answer
import numpy as np

v2_router = APIRouter(prefix="/v2")


@v2_router.post("/ask_ai")
def ask_ai(
    question: str = Query(..., description="Your question about the PDFs"),
    top_k: int = Query(3, description="Number of top chunks to use as context"),
):
    embedder = get_embedder()
    q_vec = embedder.embed([question])[0]
    conn = get_db_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                org_id = get_or_create_org("default")
                cur.execute(
                    "SELECT id FROM pdfs WHERE org_id=%s AND is_active=TRUE",
                    (org_id,),
                )
                pdf_ids = [row[0] for row in cur.fetchall()]
                if not pdf_ids:
                    return {"error": "No PDFs found for org."}
                q_vec_str = '[' + ','.join(str(float(x)) for x in q_vec) + ']'
                cur.execute(
                    """
                    SELECT c.text, p.filename, e.embedding <#> %s::vector AS distance
                    FROM chunks c
                    JOIN pdfs p ON c.pdf_id = p.id
                    JOIN embeddings e ON e.chunk_id = c.id
                    WHERE c.pdf_id = ANY(%s)
                    ORDER BY distance ASC
                    LIMIT %s
                """,
                    (q_vec_str, pdf_ids, top_k),
                )
                rows = cur.fetchall()
                if not rows:
                    return {"error": "No relevant context found."}
                context = "\n\n".join(
                    [f"[{filename}] says {text}" for text, filename, _ in rows]
                )
                sources = list(set([filename for _, filename, _ in rows]))
                prompt = f"""Context:\n{context}\n\nQuestion: {question}\nAnswer:"""
                answer = generate_llm_answer(prompt, model_name="llama3.2")
                return {
                    "answer": answer,
                    "context_chunks": [row[0] for row in rows],
                    "sources": sources,
                }
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()
