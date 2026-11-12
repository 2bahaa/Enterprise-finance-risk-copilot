
from pathlib import Path
from app.schemas import KPIDashboardResponse
from app.services.kpi_service import get_financial_kpis
from app.schemas import AnomalyDetectionResponse
from app.services.anomaly_service import detect_financial_anomalies

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import get_db
from app.core.security import verify_api_key
from app.schemas import DatasetResponse
from app.services.csv_processor import process_csv
from app.services.ingestion_service import ingest_dataset


router = APIRouter(
    prefix="/financial",
    tags=["Financial Intelligence"],
    dependencies=[Depends(verify_api_key)],
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post(
    "/datasets",
    response_model=DatasetResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    # 1. Validate filename
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is missing.",
        )

    filename = Path(file.filename).name

    # 2. Validate file extension
    if Path(filename).suffix.lower() != ".csv":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are accepted.",
        )

    # 3. Read uploaded file (maximum 10 MB)
    content = await file.read(MAX_FILE_SIZE + 1)

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds the 10 MB limit.",
        )

    try:
        # 4. Parse and validate CSV
        file_hash, transactions = process_csv(content)

        # 5. Save dataset and transactions to PostgreSQL
        dataset = ingest_dataset(
            db=db,
            filename=filename,
            file_hash=file_hash,
            transactions=transactions,
        )

        return dataset

    except ValueError as exc:
        status_code = (
            status.HTTP_409_CONFLICT
            if "already been uploaded" in str(exc)
            else status.HTTP_400_BAD_REQUEST
        )

        raise HTTPException(
            status_code=status_code,
            detail=str(exc),
        ) from exc

    except IntegrityError as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Duplicate dataset or transaction detected.",
        ) from exc

    finally:
        await file.close()

@router.get(
    "/kpis",
    response_model=KPIDashboardResponse,
)
def financial_kpis(
    dataset_id: int | None = None,
    db: Session = Depends(get_db),
):
    try:
        return get_financial_kpis(
            db=db,
            dataset_id=dataset_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
        
@router.get(
    "/anomalies",
    response_model=AnomalyDetectionResponse,
)
def financial_anomalies(
    dataset_id: int | None = None,
    db: Session = Depends(get_db),
):
    try:
        return detect_financial_anomalies(
            db=db,
            dataset_id=dataset_id,
        )

    except ValueError as exc:
        error_status = (
            status.HTTP_404_NOT_FOUND
            if str(exc) == "Dataset not found."
            else status.HTTP_422_UNPROCESSABLE_ENTITY
        )

        raise HTTPException(
            status_code=error_status,
            detail=str(exc),
        ) from exc