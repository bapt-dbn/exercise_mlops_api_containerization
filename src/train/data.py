import re
import string
from enum import StrEnum
from pathlib import Path

import pandas as pd

from src.exceptions import DataLoadError
from src.exceptions import DataNotFoundError
from src.exceptions import DataValidationError
from src.train.config import settings


class DatasetColumn(StrEnum):
    LABEL = "label"
    MESSAGE = "message"


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path or settings.data.path)

    if not path.exists():
        raise DataNotFoundError(f"Data file not found: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise DataLoadError(f"Failed to load data from {path}: {e}") from e

    required_columns = {col.value for col in DatasetColumn}
    if not required_columns.issubset(df.columns):
        raise DataValidationError(f"Dataset must contain columns: {required_columns}. Found: {list(df.columns)}")

    return df


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())

    return text  # noqa: RET504


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[DatasetColumn.MESSAGE] = df[DatasetColumn.MESSAGE].apply(preprocess_text)
    df = df.dropna(subset=[DatasetColumn.MESSAGE, DatasetColumn.LABEL])
    return df  # noqa: RET504
