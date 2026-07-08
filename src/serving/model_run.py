import os

import uvicorn
from dotenv import load_dotenv

from src.serving.model_serving import ServingState, build_serving_app

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_DATE = os.getenv("MODEL_DATE", "20260525")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", f"{MODEL_NAME}_{MODEL_DATE}")
MODEL_VERSION_ALIAS = os.getenv("MODEL_VERSION_ALIAS", "Staging")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000/")
HOST = "0.0.0.0"
PORT = 8000

state = ServingState.from_registered_model(
    tracking_uri=MLFLOW_TRACKING_URI,
    registered_model_name=REGISTERED_MODEL_NAME,
    model_version_alias=MODEL_VERSION_ALIAS,
)

app = build_serving_app(state)

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
