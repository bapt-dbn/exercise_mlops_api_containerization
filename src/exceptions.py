from enum import StrEnum
from typing import ClassVar


class OriginError(StrEnum):
    DAT = "DAT"  # Data
    FEA = "FEA"  # FeatureEngineering
    MOD = "MOD"  # Model
    EVA = "EVA"  # Evaluation
    CFG = "CFG"  # Config


class ReasonError(StrEnum):
    VAL = "VAL"  # Validation
    NOT = "NOT"  # Not Found
    INT = "INT"  # Integrity
    IO = "IO"  # I/O


class CourseMLOpsError(Exception):
    default_message: ClassVar[str | None] = None
    origin: ClassVar[str]  # Must be 3 chars (e.g., "DAT", "FEA", "MOD")
    error_type: ClassVar[str]  # Must be 3 chars (e.g., "VAL", "NOT", "INT")
    code: ClassVar[int]  # Must be < 1000

    def __init__(self, message: str | None = None) -> None:
        if message is None:
            message = self.default_message
        if message is None:
            raise ValueError(f"{self.__class__.__name__} requires a message")
        super().__init__(message)
        self.message = message

    @property
    def error_code(self) -> str:
        return f"CML-{self.origin.upper()}-{self.error_type.upper()}-{self.code:03d}"

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"


class DataValidationError(CourseMLOpsError):
    origin = OriginError.DAT
    error_type = ReasonError.VAL
    code = 1
    default_message = "Data validation failed"


class DataNotFoundError(CourseMLOpsError):
    origin = OriginError.DAT
    error_type = ReasonError.NOT
    code = 2
    default_message = "Data file not found"


class DataLoadError(CourseMLOpsError):
    origin = OriginError.DAT
    error_type = ReasonError.IO
    code = 3
    default_message = "Failed to load data"


class FeatureExtractionError(CourseMLOpsError):
    origin = OriginError.FEA
    error_type = ReasonError.VAL
    code = 1
    default_message = "Feature extraction failed"


class ModelTrainingError(CourseMLOpsError):
    origin = OriginError.MOD
    error_type = ReasonError.VAL
    code = 1
    default_message = "Model training failed"


class EvaluationError(CourseMLOpsError):
    origin = OriginError.EVA
    error_type = ReasonError.VAL
    code = 1
    default_message = "Model evaluation failed"


class ConfigurationError(CourseMLOpsError):
    origin = OriginError.CFG
    error_type = ReasonError.VAL
    code = 1
    default_message = "Configuration validation failed"


class ModelNotLoadedError(CourseMLOpsError):
    origin = OriginError.MOD
    error_type = ReasonError.NOT
    code = 2
    default_message = "Model is not loaded. Please try again later."


class ModelPredictionError(CourseMLOpsError):
    origin = OriginError.MOD
    error_type = ReasonError.INT
    code = 3
    default_message = "Prediction failed. Please try again later."


class InvalidInputError(CourseMLOpsError):
    origin = OriginError.DAT
    error_type = ReasonError.VAL
    code = 4
    default_message = "Invalid input provided."


class ModelMetadataFetchError(CourseMLOpsError):
    origin = OriginError.MOD
    error_type = ReasonError.IO
    code = 4
    default_message = "Failed to fetch model metadata from MLflow."
