import os

import uvicorn
from dotenv import load_dotenv

from src.serving.model_serving import ServingState, build_serving_app

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "svr_regression_v1_20260727")
MODEL_VERSION_ALIAS = os.getenv("MODEL_VERSION_ALIAS", "Staging")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000/")
HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "8100"))

state = ServingState.from_registered_model(
    tracking_uri=MLFLOW_TRACKING_URI,
    registered_model_name=REGISTERED_MODEL_NAME,
    model_version_alias=MODEL_VERSION_ALIAS,
)

app = build_serving_app(state)

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
