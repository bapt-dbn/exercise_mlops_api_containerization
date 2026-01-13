import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.preprocessing import LabelEncoder

from src.train.config import Settings
from src.train.data import DatasetColumn
from src.train.data import load_data
from src.train.data import preprocess_dataframe
from src.train.data import preprocess_text
from src.train.evaluation import evaluate_model
from src.train.features import combine_features
from src.train.features import create_scaler
from src.train.features import extract_numerical_features
from src.train.features import fit_scaler
from src.train.features import fit_vectorizer
from src.train.features import scale_numerical_features
from src.train.features import transform_texts
from src.train.model import create_model
from src.train.model import split_data
from src.train.model import train_model


class SpamClassifierWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, vectorizer, scaler, model):
        self.vectorizer = vectorizer
        self.scaler = scaler
        self.model = model

    def predict(self, context, model_input: pd.DataFrame) -> dict:
        texts = model_input[DatasetColumn.MESSAGE].tolist()

        texts_clean = [preprocess_text(t) for t in texts]
        X_tfidf = self.vectorizer.transform(texts_clean)
        X_numerical_scaled = scale_numerical_features(self.scaler, extract_numerical_features(texts))

        X = combine_features(X_tfidf, X_numerical_scaled)
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        return {
            "predictions": predictions,
            "probabilities": probabilities,
        }


def run_training(config_path: str | None = None) -> str:
    settings = Settings.from_yaml(config_path) if config_path else Settings.from_yaml()

    # TODO: Tracking URL and experiment name

    mlflow.sklearn.autolog(log_models=False)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"[+] Starting MLflow run: {run_id}")

        # TODO: Store parameters in MLflow

        print("[+] Loading data...")
        df = load_data(settings.data.path)
        # TODO: Store len of the data in MLFLOW before preprocessing

        print("[+] Extracting numerical features...")
        X_numerical = extract_numerical_features(df[DatasetColumn.MESSAGE].tolist())

        print("[+] Preprocessing...")
        df = preprocess_dataframe(df)
        # TODO: Store len of the data in MLFLOW after preprocessing

        class_counts = df[DatasetColumn.LABEL].value_counts().to_dict()
        for label, count in class_counts.items():
            # TODO: Store label in MLFlow

        print("[+] Fitting vectorizer and scaler...")
        vectorizer = fit_vectorizer(df[DatasetColumn.MESSAGE].tolist())
        X_tfidf = transform_texts(vectorizer, df[DatasetColumn.MESSAGE].tolist())

        scaler = create_scaler()
        scaler = fit_scaler(scaler, X_numerical)
        X_numerical_scaled = scale_numerical_features(scaler, X_numerical)

        X = combine_features(X_tfidf, X_numerical_scaled)

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df[DatasetColumn.LABEL].values)

        # TODO: Store feature information in MLFlow

        X_train, X_test, y_train, y_test = split_data(X, y)
        # TODO: Store train/test split information in MLFLOW

        print("[+] Training model...")
        model = create_model()
        model = train_model(model, X_train, y_train)

        print("[+] Evaluating...")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        metrics = evaluate_model(y_test, y_pred, y_proba)

        # TODO: Store evaluation metrics in MLFlow


        print(f"[!]   Accuracy:  {metrics.accuracy:.4f}")
        print(f"[!]   Precision: {metrics.precision:.4f}")
        print(f"[!]   Recall:    {metrics.recall:.4f}")
        print(f"[!]   F1:        {metrics.f1:.4f}")
        print(f"[!]   ROC-AUC:   {metrics.roc_auc:.4f}")

        print("[+] Logging model to MLflow...")

        wrapped_model = SpamClassifierWrapper(vectorizer, scaler, model)

        example_input = pd.DataFrame({DatasetColumn.MESSAGE: ["Free money click here!"]})
        example_output = wrapped_model.predict(None, example_input)
        signature = infer_signature(example_input, example_output)

        # TODO: Store model signature in MLFlow

        # TODO: Store model in MLflow
        print(f"[+] Model registered: {registered_model.name} v{registered_model.version}")

        if config_path:
            # TODO: Store configuration in MLflow

        print(f"[+] Training complete! Run ID: {run_id}")
        print("   View at: mlflow ui --port 5000")

        return run_id  # type: ignore[no-any-return]
