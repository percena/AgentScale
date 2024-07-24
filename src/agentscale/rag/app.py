# Refined RAG Demo with FastAPI

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import psycopg
import tiktoken
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from openai import OpenAI
from pgvector.psycopg import register_vector
from pydantic import BaseModel
from pypdf import PdfReader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set up OpenAI API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()


class Document:
    def __init__(self, content: str, metadata: Optional[Dict] = None):
        self.content = content
        self.metadata = metadata or {}


class Chunk:
    def __init__(self, content: str, metadata: Optional[Dict] = None):
        self.content = content
        self.metadata = metadata or {}


class EmbeddedChunk(Chunk):
    def __init__(
        self, content: str, embedding: List[float], metadata: Optional[Dict] = None
    ):
        super().__init__(content, metadata)
        self.embedding = embedding


class QueryResult(BaseModel):
    response: str
    source_chunks: List[Dict]


# Document Loading
def load_pdf(file_path: str) -> Document:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return Document(content=text, metadata={"source": file_path})


# Chunking
def chunk_document(
    doc: Document, chunk_size: int = 500, chunk_overlap: int = 50
) -> List[Chunk]:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(doc.content)
    chunks = []

    for i in range(0, len(tokens), chunk_size - chunk_overlap):
        chunk_tokens = tokens[i : i + chunk_size]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(Chunk(content=chunk_text, metadata=doc.metadata.copy()))

    return chunks


# Embedding
def embed_chunks(chunks: List[Chunk]) -> List[EmbeddedChunk]:
    embedded_chunks = []
    for chunk in chunks:
        embedding = get_embedding(chunk.content)
        embedded_chunks.append(
            EmbeddedChunk(
                content=chunk.content, embedding=embedding, metadata=chunk.metadata
            )
        )
    return embedded_chunks


def get_embedding(text: str) -> List[float]:
    try:
        response = client.embeddings.create(input=text, model="text-embedding-3-small")
        return response.data[0].embedding
    except Exception:
        raise


# Indexing
def index_chunks(embedded_chunks: List[EmbeddedChunk]):
    with psycopg.connect(DATABASE_URL) as conn:
        for chunk in embedded_chunks:
            timestamp = datetime.now()
            conn.execute(
                "INSERT INTO documents (content, embedding, metadata, created_at) VALUES (%s, %s, %s, %s)",
                (chunk.content, chunk.embedding, json.dumps(chunk.metadata), timestamp),
            )


# Use this DATABASE_URL when setting up your database connection
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:postgres@pgvector:5432/postgres"
)


def setup_database():
    # Connect to the 'postgres' database
    with psycopg.connect(DATABASE_URL) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            # Create the pgvector extension if it doesn't exist
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

        # Register the vector type with psycopg
        register_vector(conn)

        # Drop and create tables in the 'postgres' database directly
        conn.execute("DROP TABLE IF EXISTS documents")
        conn.execute(
            """
            CREATE TABLE documents (
                id bigserial PRIMARY KEY,
                content text,
                embedding vector(1536),
                metadata jsonb,
                created_at timestamp
            )
        """
        )
        conn.execute(
            "CREATE INDEX ON documents USING GIN (to_tsvector('english', content))"
        )


# Retrieval
def retrieve_chunks(query: str, k: int = 5) -> List[Dict]:
    embedding = get_embedding(query)

    sql = """
    WITH semantic_search AS (
        SELECT id, content, metadata, RANK () OVER (ORDER BY embedding <-> %s::vector(1536)) AS rank
        FROM documents
        ORDER BY embedding <-> %s::vector(1536)
        LIMIT %s
    ),
    keyword_search AS (
        SELECT id, content, metadata, RANK () OVER (ORDER BY ts_rank_cd(to_tsvector('english', content), query) DESC)
        FROM documents, plainto_tsquery('english', %s) query
        WHERE to_tsvector('english', content) @@ query
        ORDER BY ts_rank_cd(to_tsvector('english', content), query) DESC
        LIMIT %s
    )
    SELECT
        COALESCE(semantic_search.id, keyword_search.id) AS id,
        COALESCE(semantic_search.content, keyword_search.content) AS content,
        COALESCE(semantic_search.metadata, keyword_search.metadata) AS metadata,
        COALESCE(1.0 / (%s + semantic_search.rank), 0.0) +
        COALESCE(1.0 / (%s + keyword_search.rank), 0.0) AS score
    FROM semantic_search
    FULL OUTER JOIN keyword_search ON semantic_search.id = keyword_search.id
    ORDER BY score DESC
    LIMIT %s
    """

    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        results = conn.execute(
            sql, (embedding, embedding, k, query, k, k, k, k)
        ).fetchall()

    return [
        {"id": row[0], "content": row[1], "metadata": row[2], "score": row[3]}
        for row in results
    ]


# Generation
def generate_response(query: str, chunks: List[Dict]) -> str:
    context = "\n".join([chunk["content"] for chunk in chunks])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer the question based on the given context.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content.strip()


# Indexing Pipeline
def index_pdf(file_path: str):
    doc = load_pdf(file_path)
    chunks = chunk_document(doc)
    embedded_chunks = embed_chunks(chunks)
    index_chunks(embedded_chunks)


# Query Pipeline
def query_pipeline(query: str) -> QueryResult:
    chunks = retrieve_chunks(query)
    response = generate_response(query, chunks)
    return QueryResult(response=response, source_chunks=chunks)


# FastAPI endpoints
# Exception Handler
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"An error occurred: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500, content={"detail": "An internal server error occurred."}
    )


# Pydantic models for the request bodies
class IndexPDFRequest(BaseModel):
    file_path: str


class QueryRequest(BaseModel):
    query: str


# Index PDF Endpoint
@app.post("/index_pdf")
async def api_index_pdf(request: IndexPDFRequest):
    try:
        file_path = request.file_path
        logger.info(f"Indexing PDF: {file_path}")
        index_pdf(file_path)  # Assuming index_pdf is a defined function
        return {"message": "PDF indexed successfully"}
    except Exception as e:
        logger.error(f"Error indexing PDF: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to index PDF")


# Query Endpoint
@app.post("/query")
async def api_query(request: QueryRequest):
    try:
        query_text = request.query
        logger.info(f"Processing query: {query_text}")
        result = query_pipeline(
            query_text
        )  # Assuming query_pipeline is a defined function
        return result
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process query")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# Main function to run the server
if __name__ == "__main__":
    setup_database()
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="info")
