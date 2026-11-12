from pydantic import BaseModel, Field


class FinancialQuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)


class RAGSource(BaseModel):
    source_id: int
    document_id: str
    filename: str
    page_number: int
    relevance_score: float


class FinancialQuestionResponse(BaseModel):
    question: str
    answer: str
    sources: list[RAGSource]