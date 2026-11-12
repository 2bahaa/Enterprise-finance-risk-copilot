from fastapi import APIRouter, Depends, HTTPException, status

from app.core.security import verify_api_key
from app.schemas import CopilotRequest, CopilotResponse
from app.services.copilot_graph import run_copilot

router = APIRouter(
    prefix="/copilot",
    tags=["Unified Copilot"],
    dependencies=[Depends(verify_api_key)],
)


@router.post("/ask", response_model=CopilotResponse)
def ask_copilot(request: CopilotRequest):
    try:
        return run_copilot(
            question=request.question,
            top_k=request.top_k,
            dataset_id=request.dataset_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc