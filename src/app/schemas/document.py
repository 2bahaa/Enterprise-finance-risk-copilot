from pydantic import BaseModel


class DocumentIngestionResponse(BaseModel):
    document_id: str
    filename: str
    pages_processed: int
    chunks_created: int
    vectors_stored: int
    collection_name: str