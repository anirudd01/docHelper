from typing import Optional
import psycopg2
import os


def get_db_conn():
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


if __name__ == "__main__":
    init_tables()
