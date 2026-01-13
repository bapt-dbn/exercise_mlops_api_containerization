import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from src.exceptions import FeatureExtractionError
from src.train.config import settings


def create_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=settings.features.tfidf.max_features,
        ngram_range=settings.features.tfidf.ngram_range,
        min_df=settings.features.tfidf.min_df,
    )


def fit_vectorizer(texts: list[str]) -> TfidfVectorizer:
    vectorizer = create_vectorizer()
    vectorizer.fit(texts)
    return vectorizer


def transform_texts(vectorizer: TfidfVectorizer, texts: list[str]) -> spmatrix:
    return vectorizer.transform(texts)


def extract_numerical_features(texts: list[str]) -> np.ndarray:
    if not texts:
        raise FeatureExtractionError("Cannot extract features from empty text list")

    features = []
    for text in texts:
        text_length = len(text)
        word_count = len(text.split())
        caps_count = sum(1 for c in text if c.isupper())
        caps_ratio = caps_count / text_length if text_length > 0 else 0
        special_chars_count = sum(1 for c in text if c in "!?$€£%")

        features.append([text_length, word_count, caps_ratio, special_chars_count])

    return np.array(features)


def create_scaler() -> StandardScaler:
    return StandardScaler()


def fit_scaler(scaler: StandardScaler, X_numerical: np.ndarray) -> StandardScaler:
    return scaler.fit(X_numerical)


def scale_numerical_features(scaler: StandardScaler, X_numerical: np.ndarray) -> np.ndarray:
    return scaler.transform(X_numerical)


def combine_features(X_tfidf: spmatrix, X_numerical_scaled: np.ndarray) -> spmatrix:
    return hstack([X_tfidf, csr_matrix(X_numerical_scaled)])
