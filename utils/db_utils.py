import os
from typing import Optional

import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_db_conn():
    conn_str = os.getenv("DATABASE_URL")
    if conn_str:
        return psycopg2.connect(conn_str)
    return psycopg2.connect(
        dbname=os.getenv("PGDATABASE", "postgres"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", "password"),
        host=os.getenv("PGHOST", "0.0.0.0"),
        port=os.getenv("PGPORT", "15432"),
    )


def execute_sql(sql, params=None):
    conn = get_db_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
    finally:
        conn.close()


def init_tables():
    execute_sql("""CREATE EXTENSION IF NOT EXISTS vector;""")
    sql = """
    CREATE TABLE IF NOT EXISTS orgs (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        is_active BOOLEAN DEFAULT TRUE
    );
    CREATE TABLE IF NOT EXISTS pdfs (
        id SERIAL PRIMARY KEY,
        org_id INTEGER REFERENCES orgs(id),
        filename TEXT NOT NULL,
        chunk_size INTEGER NOT NULL,
        upload_time TIMESTAMP DEFAULT NOW(),
        is_active BOOLEAN DEFAULT TRUE
    );
    CREATE TABLE IF NOT EXISTS chunks (
        id SERIAL PRIMARY KEY,
        pdf_id INTEGER REFERENCES pdfs(id),
        chunk_index INTEGER NOT NULL,
        text TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS embeddings (
        id SERIAL PRIMARY KEY,
        chunk_id INTEGER REFERENCES chunks(id),
        embedding VECTOR(384) -- adjust dimension as needed
    );
    """
    execute_sql(sql)
    execute_sql("INSERT INTO orgs (name) VALUES (%s)", ("default",))


def get_or_create_org(name: str = "default") -> Optional[int]:
    conn = get_db_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM orgs WHERE name=%s AND is_active=TRUE", (name,)
                )
                row = cur.fetchone()
                if row:
                    return row[0]
                else:
                    execute_sql(
                        "INSERT INTO orgs (name) VALUES (%s) RETURNING id", (name,)
                    )
                    return cur.fetchone()[0]
    finally:
        conn.close()


def insert_pdf(org_id: int, filename: str, chunk_size: int) -> int:
    conn = get_db_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO pdfs (org_id, filename, chunk_size) VALUES (%s, %s, %s) RETURNING id",
                    (org_id, filename, chunk_size),
                )
                return cur.fetchone()[0]
    finally:
        conn.close()


def insert_chunk(pdf_id: int, chunk_index: int, text: str) -> int:
    conn = get_db_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chunks (pdf_id, chunk_index, text) VALUES (%s, %s, %s) RETURNING id",
                    (pdf_id, chunk_index, text),
                )
                return cur.fetchone()[0]
    finally:
        conn.close()


def insert_embedding(chunk_id: int, embedding: list) -> int:
    conn = get_db_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO embeddings (chunk_id, embedding) VALUES (%s, %s) RETURNING id",
                    (chunk_id, embedding),
                )
                return cur.fetchone()[0]
    finally:
        conn.close()


def get_pdf_id_by_filename(filename: str, org_id: int = None) -> Optional[int]:
    """Get PDF ID by filename and optionally org_id."""
    conn = get_db_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                if org_id:
                    cur.execute(
                        "SELECT id FROM pdfs WHERE filename=%s AND org_id=%s AND is_active=TRUE",
                        (filename, org_id),
                    )
                else:
                    cur.execute(
                        "SELECT id FROM pdfs WHERE filename=%s AND is_active=TRUE",
                        (filename,),
                    )
                row = cur.fetchone()
                return row[0] if row else None
    finally:
        conn.close()


def remove_pdf_data(filename: str, org_id: int = None) -> dict:
    """
    Remove PDF data from database.

    Args:
        filename: Name of the PDF file
        org_id: Optional org ID to filter by
        soft_delete: If True, mark as inactive. If False, delete records completely.

    Returns:
        dict: Status of the operation
    """
    conn = get_db_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                # Get PDF ID
                pdf_id = get_pdf_id_by_filename(filename, org_id)
                if not pdf_id:
                    return {"success": False, "error": "PDF not found in database"}

                # Hard delete: Remove embeddings first (due to foreign key constraint)
                cur.execute(
                    "DELETE FROM embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE pdf_id=%s)",
                    (pdf_id,),
                )

                # Remove chunks
                cur.execute("DELETE FROM chunks WHERE pdf_id=%s", (pdf_id,))

                cur.execute("UPDATE pdfs SET is_active=FALSE WHERE id=%s", (pdf_id,))
                return {
                    "success": True,
                    "message": f"PDF '{filename}' and all associated data removed",
                    "pdf_id": pdf_id,
                }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        conn.close()


def clean_existing_chunks_batch(batch_size: int = 10) -> dict:
    """
    Clean existing chunks in the database in batches to handle large datasets efficiently.
    Also regenerates vectors to match the cleaned chunks.

    Args:
        batch_size: Number of chunks to process in each batch

    Returns:
        dict: Status of the operation with count of cleaned chunks and regenerated vectors
    """
    from utils.text_cleaner import TextCleaner
    from utils.vector_utils import get_embedder

    conn = get_db_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                # Get total count first
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM chunks c 
                    JOIN pdfs p ON c.pdf_id = p.id 
                    WHERE c.text IS NOT NULL AND p.is_active = TRUE
                """)
                total_chunks = cur.fetchone()[0]

                cleaned_count = 0
                vector_regenerated_count = 0
                embedder = get_embedder()

                # Process in batches
                offset = 0
                while offset < total_chunks:
                    # Get batch of chunks
                    cur.execute(
                        """
                        SELECT c.id, c.text, c.pdf_id, p.filename 
                        FROM chunks c 
                        JOIN pdfs p ON c.pdf_id = p.id 
                        WHERE c.text IS NOT NULL AND p.is_active = TRUE
                        ORDER BY c.id
                        LIMIT %s OFFSET %s
                    """,
                        (batch_size, offset),
                    )

                    batch_chunks = cur.fetchall()
                    if not batch_chunks:
                        break

                    # Process batch
                    for chunk_id, text, pdf_id, filename in batch_chunks:
                        if text:
                            # Clean the text
                            cleaned_text = TextCleaner.clean_text_aggressive(text)

                            # Only update if the text actually changed and is not empty
                            if cleaned_text != text and cleaned_text.strip():
                                # Update the chunk text
                                cur.execute(
                                    "UPDATE chunks SET text = %s WHERE id = %s",
                                    (cleaned_text, chunk_id),
                                )
                                cleaned_count += 1

                                # Regenerate vector for the cleaned chunk
                                try:
                                    # Generate new embedding
                                    new_vector = embedder.embed([cleaned_text])[0]

                                    # Update the embedding in the database
                                    cur.execute(
                                        "UPDATE embeddings SET embedding = %s WHERE chunk_id = %s",
                                        (new_vector, chunk_id),
                                    )
                                    vector_regenerated_count += 1

                                except Exception as e:
                                    print(
                                        f"Error regenerating vector for chunk {chunk_id}: {e}"
                                    )
                                    # Continue with other chunks even if one fails

                    offset += batch_size
                    print(
                        f"Processed batch: {min(offset, total_chunks)}/{total_chunks} chunks"
                    )

                return {
                    "success": True,
                    "message": f"Cleaned {cleaned_count} chunks and regenerated {vector_regenerated_count} vectors in database (batch processing)",
                    "chunks_cleaned": cleaned_count,
                    "vectors_regenerated": vector_regenerated_count,
                    "total_chunks": total_chunks,
                    "batch_size": batch_size,
                }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        conn.close()


if __name__ == "__main__":
    init_tables()
