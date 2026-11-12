from pydantic import BaseModel, ConfigDict, Field

from app.schemas.rag import RAGSource


class CopilotRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "question": "Which department spends the most "
                                "and are there suspicious transactions?",
                    "top_k": 3,
                }
            ]
        }
    )

    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)
    # Optional. Leave it out to analyse all datasets.
    dataset_id: int | None = Field(default=None, ge=1)


class CopilotResponse(BaseModel):
    question: str
    answer: str
    routes_used: list[str]
    sources: list[RAGSource]
    warnings: list[str]