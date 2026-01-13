import numpy as np
from scipy.sparse import spmatrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.exceptions import ModelTrainingError
from src.train.config import settings


def create_model() -> LogisticRegression:
    return LogisticRegression(
        C=settings.model.params.C,
        max_iter=settings.model.params.max_iter,
        solver=settings.model.params.solver,
    )


def split_data(
    X: spmatrix,
    y: np.ndarray,
) -> tuple[spmatrix, spmatrix, np.ndarray, np.ndarray]:
    return train_test_split(  # type: ignore[no-any-return]
        X,
        y,
        test_size=settings.data.test_size,
        random_state=settings.data.random_state,
        stratify=y,
    )


def train_model(
    model: LogisticRegression,
    X_train: spmatrix,
    y_train: np.ndarray,
) -> LogisticRegression:
    if X_train.shape[0] == 0:
        raise ModelTrainingError("Cannot train model with empty training data")

    try:
        model.fit(X_train, y_train)
    except Exception as e:
        raise ModelTrainingError(f"Model training failed: {e}") from e

    return model
