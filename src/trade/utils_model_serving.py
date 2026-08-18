import logging
import os

import pandas as pd
import requests
from dotenv import load_dotenv

PREDICT_URL = os.getenv("MODEL_PREDICT_URL", "http://localhost:8100/predict")
MODEL_PARAMS_URL = os.getenv("MODEL_PARAMS_URL", "http://localhost:8100/model/params")
MODEL_INFO_URL = os.getenv("MODEL_INFO_URL", "http://localhost:8100/")


load_dotenv()

def request_predictions(X: pd.DataFrame, n_rows: int, logger: logging.Logger) -> dict:
    """POST the last n rows of X to the model serving /predict endpoint."""
    sample = X.tail(n_rows)
    payload = {
        "columns": sample.columns.tolist(),
        "data": sample.values.tolist(),
    }
    try:
        response = requests.post(
            PREDICT_URL,
            json=payload
        )
        return response.json()
    except requests.exceptions.RequestException as exc:
        logger.error("Prediction request failed: %s", exc)
        raise


def fetch_model_params(logger: logging.Logger) -> dict:
    """Fetch model_params from the model serving endpoint."""
    response = requests.get(MODEL_PARAMS_URL)
    try:
        return response.json()
    except requests.exceptions.RequestException as exc:
        logger.error("Model params request failed: %s", exc)
        raise

def fetch_model_info(logger: logging.Logger) -> dict:
    """Fetch model info from the model serving endpoint."""
    response = requests.get(MODEL_INFO_URL)
    try:
        return response.json()
    except requests.exceptions.RequestException as exc:
        logger.error("Model info request failed: %s", exc)
        raise
