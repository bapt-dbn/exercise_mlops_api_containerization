from src.utils import EnvironmentVariable

API_TITLE = "Spam Classifier API"
API_VERSION = "0.2.0"
MODEL_URI = "models:/spam-classifier/latest"

MLFLOW_UI_BASE = EnvironmentVariable.MLFLOW_UI_BASE.read(default="http://localhost:5000")
API_HOST = EnvironmentVariable.API_HOST.read(default="0.0.0.0")
API_PORT = int(EnvironmentVariable.API_PORT.read(default="8000"))
