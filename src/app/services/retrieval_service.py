
from app.services.embedding_service import generate_embeddings
from app.services.vector_store import (
    COLLECTION_NAME,
    get_qdrant_client,
)


def search_documents(
    question: str,
    top_k: int = 5,
) -> dict:
    """Retrieve the most relevant document chunks for a question."""

    question = question.strip()

    if not question:
        raise ValueError("Search question cannot be empty.")

    if top_k < 1 or top_k > 20:
        raise ValueError("top_k must be between 1 and 20.")

    # 1. Embed the user's question using the same model
    # that was used to embed the document chunks.
    embeddings = generate_embeddings([question])
    query_vector = embeddings[0]

    # 2. Search Qdrant for similar vectors.
    client = get_qdrant_client()

    search_response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    # 3. Format the retrieved results.
    results = []

    for point in search_response.points:
        payload = point.payload or {}

        results.append({
            "document_id": payload.get("document_id", ""),
            "filename": payload.get("filename", ""),
            "page_number": payload.get("page_number", 0),
            "chunk_index": payload.get("chunk_index", 0),
            "text": payload.get("text", ""),
            "relevance_score": round(float(point.score), 6),
        })

    return {
        "question": question,
        "results_count": len(results),
        "results": results,
    }
