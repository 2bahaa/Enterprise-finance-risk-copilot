
from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class TransactionResponse(BaseModel):
    id: int
    transaction_id: str
    transaction_date: date
    department: str
    category: str
    amount: Decimal
    currency: str
    description: str | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DatasetResponse(BaseModel):
    id: int
    filename: str
    file_hash: str
    row_count: int
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class CurrencyTotal(BaseModel):
    currency: str
    total_amount: Decimal


class DepartmentSpending(BaseModel):
    department: str
    currency: str
    total_amount: Decimal


class CategorySpending(BaseModel):
    category: str
    currency: str
    total_amount: Decimal


class MonthlySpending(BaseModel):
    month: datetime
    currency: str
    total_amount: Decimal


class KPIDashboardResponse(BaseModel):
    dataset_id: int | None
    transaction_count: int
    total_expenses: list[CurrencyTotal]
    department_spending: list[DepartmentSpending]
    category_spending: list[CategorySpending]
    monthly_trends: list[MonthlySpending]
    
class TransactionAnomaly(BaseModel):
    transaction_id: str
    transaction_date: date
    department: str
    category: str
    amount: Decimal
    currency: str
    anomaly_score: float
    is_anomaly: bool
    classification: str


class AnomalyDetectionResponse(BaseModel):
    dataset_id: int | None
    analyzed_transactions: int
    anomaly_count: int
    anomalies: list[TransactionAnomaly]