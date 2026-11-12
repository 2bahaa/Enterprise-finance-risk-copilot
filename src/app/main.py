
from fastapi import Depends, FastAPI
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.core.security import verify_api_key

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from app.database import Base, engine
from app import models
from app.api.documents import router as documents_router
from app.api.financial import router as financial_router
from app.api.copilot import router as copilot_router
@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield

app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered financial intelligence and risk analysis.",
    version="0.1.0",
    lifespan=lifespan
)
app.include_router(financial_router)
app.include_router(documents_router)
app.include_router(copilot_router)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/ui/")
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": settings.APP_NAME
    }


@app.get(
    "/health/database",
    dependencies=[Depends(verify_api_key)]
)
def database_health(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))

    return {
        "status": "healthy",
        "database": "connected"
    }
