import os

import uvicorn
from dotenv import load_dotenv

from src.serving.model_serving import build_serving_app, load_registered_model

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

MODEL_NAME = os.getenv("MODEL_NAME")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", f"{MODEL_NAME}_20260525")
MODEL_VERSION_ALIAS = os.getenv("MODEL_VERSION_ALIAS", "Staging")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
HOST = "0.0.0.0"
PORT = 8000

model, model_info, model_uri, model_version_info, model_params, example_data = load_registered_model(
    tracking_uri=MLFLOW_TRACKING_URI,
    registered_model_name=REGISTERED_MODEL_NAME,
    model_version_alias=MODEL_VERSION_ALIAS,
)

app = build_serving_app(
    tracking_uri=MLFLOW_TRACKING_URI,
    model=model,
    model_info=model_info,
    model_uri=model_uri,
    registered_model_name=REGISTERED_MODEL_NAME,
    model_version_info=model_version_info,
    model_params=model_params,
    example_data=example_data,
)

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
