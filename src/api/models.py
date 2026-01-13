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
    message: Annotated[
        str,
        Field(
            min_length=1,
            max_length=5000,
            description="Message text",
        ),
    ]


class PredictOutput(BaseModel):
    message: str
    prediction: PredictionEnum
    probability: Annotated[float, Field(ge=0, le=1, description="Spam probability")]
    confidence: ConfidenceEnum
    is_spam: bool

    @staticmethod
    def calculate_confidence(
        prob: float,
        high_min: float = 0.9,
        high_max: float = 0.1,
        medium_min: float = 0.7,
        medium_max: float = 0.3,
    ) -> ConfidenceEnum:
        if prob >= high_min or prob <= high_max:
            return ConfidenceEnum.HIGH
        if prob >= medium_min or prob <= medium_max:
            return ConfidenceEnum.MEDIUM
        return ConfidenceEnum.LOW


class HealthOutput(BaseModel):
    status: HealthStatusEnum
    model_loaded: bool
    model_uri: str | None = None
    timestamp: str


class ModelInfoOutput(BaseModel):
    model_name: str
    model_version: str | None = None
    run_id: str | None = None
    mlflow_ui_url: str
    artifact_uri: str | None = None
    registered_at: int | None = None


class ConfigOutput(BaseModel):
    model_type: str
    tfidf_max_features: int
    tfidf_ngram_range: tuple[int, int]
    tfidf_min_df: int
    numerical_features: list[str]


class ErrorOutput(BaseModel):
    detail: str
    error_code: str | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
