import hashlib
from uuid import uuid4

from app.services.document_processor import process_pdf
from app.services.embedding_service import generate_embeddings
from app.services.vector_store import (
    COLLECTION_NAME,
    document_exists_by_hash,
    store_document_chunks,
)


class DuplicateDocumentError(Exception):
    """Raised when the same PDF has already been ingested."""


def ingest_document(
    file_content: bytes,
    filename: str,
) -> dict:
    """Extract, chunk, embed, and store a PDF document."""

    # 1. Fingerprint the file and reject duplicates early,
    #    before spending time on extraction and embeddings.
    file_hash = hashlib.sha256(file_content).hexdigest()

    if document_exists_by_hash(file_hash):
        raise DuplicateDocumentError(
            "This document has already been uploaded."
        )

    # 2. Extract PDF text and create chunks.
    chunks = process_pdf(file_content)

    # 3. Generate embeddings.
    texts = [chunk["text"] for chunk in chunks]
    embeddings = generate_embeddings(texts)

    # 4. Assign a unique document ID.
    document_id = str(uuid4())

    # 5. Store vectors and metadata in Qdrant.
    stored_count = store_document_chunks(
        chunks=chunks,
        embeddings=embeddings,
        document_id=document_id,
        filename=filename,
        file_hash=file_hash,
    )

    return {
        "document_id": document_id,
        "filename": filename,
        "pages_processed": len(
            {chunk["page_number"] for chunk in chunks}
        ),
        "chunks_created": len(chunks),
        "vectors_stored": stored_count,
        "collection_name": COLLECTION_NAME,
    }