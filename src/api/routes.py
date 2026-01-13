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
    ...,
) -> PredictOutput:
    ...



@router.get("/health", response_model=HealthOutput, status_code=status.HTTP_200_OK)
async def health(request: Request) -> HealthOutput:
    ...


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
    ...
