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

    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.set_experiment(settings.mlflow.experiment_name)

    mlflow.sklearn.autolog(log_models=False)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"[+] Starting MLflow run: {run_id}")

        mlflow.log_params(
            {
                "data.path": settings.data.path,
                "data.test_size": settings.data.test_size,
                "data.random_state": settings.data.random_state,
                "features.tfidf.max_features": settings.features.tfidf.max_features,
                "features.tfidf.ngram_range": str(settings.features.tfidf.ngram_range),
                "features.tfidf.min_df": settings.features.tfidf.min_df,
                "model.type": settings.model.type,
                "model.C": settings.model.params.C,
                "model.max_iter": settings.model.params.max_iter,
            }
        )

        print("[+] Loading data...")
        df = load_data(settings.data.path)
        mlflow.log_metric("data.raw_samples", len(df))

        print("[+] Extracting numerical features...")
        X_numerical = extract_numerical_features(df[DatasetColumn.MESSAGE].tolist())

        print("[+] Preprocessing...")
        df = preprocess_dataframe(df)
        mlflow.log_metric("data.clean_samples", len(df))

        class_counts = df[DatasetColumn.LABEL].value_counts().to_dict()
        for label, count in class_counts.items():
            mlflow.log_metric(f"data.class_{label}", count)

        print("[+] Fitting vectorizer and scaler...")
        vectorizer = fit_vectorizer(df[DatasetColumn.MESSAGE].tolist())
        X_tfidf = transform_texts(vectorizer, df[DatasetColumn.MESSAGE].tolist())

        scaler = create_scaler()
        scaler = fit_scaler(scaler, X_numerical)
        X_numerical_scaled = scale_numerical_features(scaler, X_numerical)

        X = combine_features(X_tfidf, X_numerical_scaled)

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df[DatasetColumn.LABEL].values)

        mlflow.log_metric("features.vocabulary_size", len(vectorizer.vocabulary_))
        mlflow.log_metric("features.numerical_features_count", X_numerical_scaled.shape[1])

        X_train, X_test, y_train, y_test = split_data(X, y)
        mlflow.log_metric("data.train_samples", X_train.shape[0])
        mlflow.log_metric("data.test_samples", X_test.shape[0])

        print("[+] Training model...")
        model = create_model()
        model = train_model(model, X_train, y_train)

        print("[+] Evaluating...")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        metrics = evaluate_model(y_test, y_pred, y_proba)

        mlflow.log_metrics(metrics.to_dict())

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

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=wrapped_model,
            signature=signature,
            input_example=example_input,
            pip_requirements=[
                "scikit-learn>=1.5.0",
                "pandas>=2.2.0",
            ],
        )

        registered_model = mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name=settings.mlflow.registered_model_name,
        )
        print(f"[+] Model registered: {registered_model.name} v{registered_model.version}")

        if config_path:
            mlflow.log_artifact(config_path, artifact_path="config")

        print(f"[+] Training complete! Run ID: {run_id}")
        print("   View at: mlflow ui --port 5000")

        return run_id  # type: ignore[no-any-return]
