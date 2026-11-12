from pydantic import BaseModel, Field


class DocumentSearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievedChunk(BaseModel):
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    relevance_score: float


class DocumentSearchResponse(BaseModel):
    question: str
    results_count: int
    results: list[RetrievedChunk]