import os
from enum import StrEnum
from typing import Self


class EnvironmentVariable(StrEnum):
    MLFLOW_TRACKING_URI = "MLFLOW_TRACKING_URI"
    MLFLOW_TRACKING_URI_CML = "CML_MLFLOW_TRACKING_URI"
    MLFLOW_ARTIFACT_ROOT = "CML_MLFLOW_ARTIFACT_ROOT"
    MLFLOW_UI_BASE = "CML_MLFLOW_UI_BASE"
    API_HOST = "CML_API_HOST"
    API_PORT = "CML_API_PORT"
    LOG_LEVEL = "CML_LOG_LEVEL"

    def read(self: Self, default: str | None = None) -> str:
        try:
            return os.environ[self.value]
        except KeyError as e:
            if default is None:
                raise ValueError(
                    f"Environment variable {self.value} is not defined and no default value was provided"
                ) from e
            return default
