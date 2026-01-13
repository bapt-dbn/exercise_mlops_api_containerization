from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Self

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel

from src.exceptions import ConfigurationError


class DataConfig(BaseModel):
    path: str = "data/spam.csv"
    test_size: float = 0.2
    random_state: int = 42


class TfidfConfig(BaseModel):
    max_features: int = 5000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95


class FeaturesConfig(BaseModel):
    tfidf: TfidfConfig = TfidfConfig()
    numerical: list[str] = [
        "text_length",
        "word_count",
        "caps_ratio",
        "special_chars_count",
    ]


class ModelType(StrEnum):
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"


class ModelParams(BaseModel):
    C: float = 0.5
    max_iter: int = 1000
    solver: str = "liblinear"
    penalty: str = "l2"


class ModelConfig(BaseModel):
    type: ModelType = ModelType.LOGISTIC_REGRESSION
    params: ModelParams = ModelParams()


class MLflowConfig(BaseModel):
    tracking_uri: str = "mlruns"
    experiment_name: str = "spam-classifier"
    registered_model_name: str = "spam-classifier"


class ConfidenceThresholds(BaseModel):
    high_min: float = 0.9
    high_max: float = 0.1
    medium_min: float = 0.7
    medium_max: float = 0.3


class Settings(BaseModel):
    data: DataConfig = DataConfig()
    features: FeaturesConfig = FeaturesConfig()
    model: ModelConfig = ModelConfig()
    mlflow: MLflowConfig = MLflowConfig()
    confidence_thresholds: ConfidenceThresholds = ConfidenceThresholds()

    @classmethod
    def from_yaml(cls, path: str | Path = "config/config.yaml") -> Self:
        path = Path(path)
        if not path.exists():
            return cls()

        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"YAML syntax error in {path}: {e}") from e

        return cls(**data)


@lru_cache
def get_settings() -> Settings:
    return Settings.from_yaml()


settings = get_settings()
