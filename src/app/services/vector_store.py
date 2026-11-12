import os
from uuid import uuid4

from qdrant_client import QdrantClient, models
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333",
)

COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "financial_documents",
)

VECTOR_SIZE = 384

_client = None


def get_qdrant_client() -> QdrantClient:
    """Create and reuse the Qdrant client."""

    global _client

    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)

    return _client


def ensure_collection() -> None:
    """Create the collection if it does not already exist."""

    client = get_qdrant_client()

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )


def document_exists_by_hash(file_hash: str) -> bool:
    """Return True if a document with this file hash is already stored."""

    client = get_qdrant_client()

    # A brand-new system has no collection yet, so nothing can exist.
    if not client.collection_exists(COLLECTION_NAME):
        return False

    result = client.count(
        collection_name=COLLECTION_NAME,
        count_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="file_hash",
                    match=models.MatchValue(value=file_hash),
                )
            ]
        ),
        exact=True,
    )

    return result.count > 0


def store_document_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
    document_id: str,
    filename: str,
    file_hash: str,
) -> int:
    """Store document vectors and metadata in Qdrant."""

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Number of chunks must match number of embeddings."
        )

    if not chunks:
        return 0

    ensure_collection()
    client = get_qdrant_client()

    points = []

    for chunk, embedding in zip(chunks, embeddings):
        points.append(
            models.PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload={
                    "document_id": document_id,
                    "filename": filename,
                    "file_hash": file_hash,
                    "page_number": chunk["page_number"],
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                },
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )

    return len(points)