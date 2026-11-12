
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

MODEL_CACHE_DIR = Path(
    os.getenv("MODEL_CACHE_DIR", "E:/finance-copilot-models")
)

_model = None


def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it."""

    global _model

    if _model is None:
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        _model = SentenceTransformer(
            MODEL_NAME,
            device="cpu",
            cache_folder=str(MODEL_CACHE_DIR),
        )

    return _model


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Convert text chunks into normalized embedding vectors."""

    if not texts:
        return []

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings.tolist()
