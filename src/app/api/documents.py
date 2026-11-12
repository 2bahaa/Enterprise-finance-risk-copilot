from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)

from app.core.security import verify_api_key
from app.schemas import (
    DocumentIngestionResponse,
    DocumentSearchRequest,
    DocumentSearchResponse,
    FinancialQuestionRequest,
    FinancialQuestionResponse,
)
from app.services.document_ingestion_service import (
    DuplicateDocumentError,
    ingest_document,
)
from app.services.rag_service import answer_financial_question
from app.services.retrieval_service import search_documents

router = APIRouter(
    prefix="/documents",
    tags=["Document Intelligence"],
    dependencies=[Depends(verify_api_key)],
)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


@router.post(
    "/upload",
    response_model=DocumentIngestionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(...),
):
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is missing.",
        )

    filename = Path(
        file.filename.replace("\\", "/")
    ).name

    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted.",
        )

    content = await file.read(MAX_FILE_SIZE + 1)

    try:
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File exceeds the 20 MB limit.",
            )

        if not content.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is not a valid PDF.",
            )

        result = ingest_document(
            file_content=content,
            filename=filename,
        )

        return result

    except DuplicateDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    finally:
        await file.close()


@router.post(
    "/search",
    response_model=DocumentSearchResponse,
)
def search_financial_documents(
    request: DocumentSearchRequest,
):
    try:
        return search_documents(
            question=request.question,
            top_k=request.top_k,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.post("/ask", response_model=FinancialQuestionResponse)
def ask_financial_question(request: FinancialQuestionRequest):
    try:
        return answer_financial_question(
            question=request.question,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc