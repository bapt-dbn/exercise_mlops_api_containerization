from datetime import UTC
from datetime import datetime
from typing import Annotated
from typing import cast

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Request
from fastapi import status

from src.api.config import MLFLOW_UI_BASE
from src.api.dal import SpamClassifierService
from src.api.models import ConfigOutput
from src.api.models import HealthOutput
from src.api.models import HealthStatusEnum
from src.api.models import ModelInfoOutput
from src.api.models import PredictInput
from src.api.models import PredictionEnum
from src.api.models import PredictOutput
from src.exceptions import ModelNotLoadedError
from src.train.config import get_settings


def get_classifier_service(request: Request) -> SpamClassifierService:
    if not hasattr(request.app.state, "classifier_service"):
        raise ModelNotLoadedError("Classifier service not initialized")
    return cast(SpamClassifierService, request.app.state.classifier_service)


ClassifierServiceDep = Annotated[SpamClassifierService, Depends(get_classifier_service)]

router = APIRouter(prefix="/api/v1", tags=["predictions"])


@router.post("/predict", response_model=PredictOutput, status_code=status.HTTP_200_OK)
async def predict(
    classifier: ClassifierServiceDep,
    request: PredictInput,
) -> PredictOutput:
    settings = get_settings()
    thresholds = settings.confidence_thresholds
    result = classifier.predict(request.message)
    confidence = PredictOutput.calculate_confidence(
        result["probability"],
        high_min=thresholds.high_min,
        high_max=thresholds.high_max,
        medium_min=thresholds.medium_min,
        medium_max=thresholds.medium_max,
    )

    return PredictOutput(
        message=request.message,
        prediction=PredictionEnum(result["prediction"]),
        probability=result["probability"],
        confidence=confidence,
        is_spam=result["is_spam"],
    )


@router.get("/health", response_model=HealthOutput, status_code=status.HTTP_200_OK)
async def health(request: Request) -> HealthOutput:
    classifier: SpamClassifierService | None = getattr(request.app.state, "classifier_service", None)
    is_loaded = classifier is not None and classifier.is_loaded
    model_uri = classifier.model_uri if classifier is not None else None

    return HealthOutput(
        status=HealthStatusEnum.HEALTHY if is_loaded else HealthStatusEnum.DEGRADED,
        model_loaded=is_loaded,
        model_uri=model_uri,
        timestamp=datetime.now(UTC).isoformat(),
    )


@router.get("/model/info", response_model=ModelInfoOutput, status_code=status.HTTP_200_OK)
async def model_info(classifier: ClassifierServiceDep) -> ModelInfoOutput:
    return ModelInfoOutput(
        model_name="spam-classifier",
        model_version=classifier.model_version,
        run_id=classifier.run_id,
        mlflow_ui_url=f"{MLFLOW_UI_BASE}/#/runs/{classifier.run_id}" if classifier.run_id else MLFLOW_UI_BASE,
        artifact_uri=classifier.artifact_uri,
        registered_at=classifier.registered_at,
    )


@router.get("/config", response_model=ConfigOutput, status_code=status.HTTP_200_OK)
async def get_config() -> ConfigOutput:
    settings = get_settings()

    return ConfigOutput(
        model_type=settings.model.type,
        tfidf_max_features=settings.features.tfidf.max_features,
        tfidf_ngram_range=settings.features.tfidf.ngram_range,
        tfidf_min_df=settings.features.tfidf.min_df,
        numerical_features=["text_length", "word_count", "caps_ratio", "special_chars_count"],
    )
