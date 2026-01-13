from datetime import UTC
from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel
from pydantic import Field


class PredictionEnum(StrEnum):
    SPAM = "spam"
    HAM = "ham"


class ConfidenceEnum(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class HealthStatusEnum(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"


class PredictInput(BaseModel):
    ...


class PredictOutput(BaseModel):
    ...


class HealthOutput(BaseModel):
    ...


class ModelInfoOutput(BaseModel):
    model_name: str
    model_version: str | None = None
    run_id: str | None = None
    mlflow_ui_url: str
    artifact_uri: str | None = None
    registered_at: int | None = None


class ConfigOutput(BaseModel):
    ...


class ErrorOutput(BaseModel):
    detail: str
    error_code: str | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
