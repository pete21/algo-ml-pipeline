import json
import os
from typing import Any, Optional, Type

import mlflow
import pandas as pd
import pydantic
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from mlflow.models import get_model_info
from pydantic import BaseModel, Field, create_model
from xgboost import DMatrix

MLFLOW_TYPE_MAP = {
    "boolean": bool,
    "integer": int,
    "long": int,
    "float": float,
    "double": float,
    "string": str,
    "binary": bytes,
    "datetime": str,
}


def create_pydantic_model_from_signature(
    schema: mlflow.types.schema.Schema,
    model_name: str,
) -> Type[BaseModel]:
    """Build a Pydantic model from an MLflow input schema for OpenAPI/Swagger docs."""
    if schema is None or len(schema.inputs) == 0:
        raise ValueError("The model does not have an input signature defined.")

    fields_spec: dict[str, Any] = {}
    for column in schema.inputs:
        mlflow_type = column.type.name if hasattr(column.type, "name") else str(column.type)
        python_type = MLFLOW_TYPE_MAP.get(mlflow_type, float)
        fields_spec[column.name] = (
            python_type,
            Field(..., description=f"MLflow type: {mlflow_type}"),
        )

    return create_model(model_name, **fields_spec)


def load_registered_model(
    tracking_uri: str,
    registered_model_name: str,
    model_version_alias: str,
) -> tuple[Any, Any, str, Any, Optional[dict]]:
    """Load a registered MLflow XGBoost model and its metadata."""
    client = MlflowClient(tracking_uri=tracking_uri)
    model_version_info = client.get_model_version_by_alias(
        registered_model_name,
        model_version_alias,
    )
    model_uri = model_version_info.source
    model_info = get_model_info(model_uri)
    model = mlflow.xgboost.load_model(model_uri)

    example_data = None
    if model_info.saved_input_example_info:
        artifact_path = model_info.saved_input_example_info["artifact_path"]
        model_name = os.getenv("MODEL_NAME", "model")
        path = f"{model_name}/{artifact_path}"
        local_path = client.download_artifacts(
            run_id=model_version_info.run_id,
            path=path,
        )
        with open(local_path, "r", encoding="utf-8") as file:
            example_data = json.load(file)

    return model, model_info, model_uri, model_version_info, example_data


def build_serving_app(
    model: Any,
    model_info: Any,
    model_uri: str,
    registered_model_name: str,
    model_version_info: Any,
    example_data: Optional[dict] = None,
) -> FastAPI:
    """Create a FastAPI app with Swagger docs derived from the MLflow input signature."""
    if model_info.signature is None or model_info.signature.inputs is None:
        raise ValueError("Model signature inputs are required for serving.")

    input_model = create_pydantic_model_from_signature(
        model_info.signature.inputs,
        "ModelInput",
    )
    batch_input_model = create_model(
        "BatchModelInput",
        inputs=(list[input_model], Field(..., description="Batch of model inputs")),
    )
    input_names = [column.name for column in model_info.signature.inputs]

    app = FastAPI(
        title=f"MLflow Model Serving: {registered_model_name}",
        description=(
            "Prediction API with request schema generated from "
            "`model_info.signature.inputs`."
        ),
        version=str(model_version_info.version),
    )

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "message": "MLflow model prediction API",
            "model_uri": model_uri,
            "registered_model_name": registered_model_name,
            "model_version": model_version_info.version,
            "model_version_alias": list(model_version_info.aliases),
            "input_features": len(input_names),
            "docs_url": "/docs",
            "openapi_url": "/openapi.json",
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/model/signature")
    def model_signature() -> dict[str, Any]:
        return {
            "inputs": model_info.signature.inputs.to_dict(),
            "outputs": model_info.signature.outputs.to_dict()
            if model_info.signature.outputs
            else None,
        }

    def _predict_dataframe(df: pd.DataFrame) -> list[list[float]]:
        missing_columns = set(input_names) - set(df.columns)
        if missing_columns:
            raise HTTPException(
                status_code=422,
                detail=f"Missing input features: {sorted(missing_columns)}",
            )
        ordered_df = df[input_names]
        predictions = model.predict(DMatrix(ordered_df))
        return predictions.tolist()

    @app.post("/predict")
    def predict(model_input: input_model) -> dict[str, Any]:
        row = pd.DataFrame([model_input.model_dump()], columns=input_names)
        return {"predictions": _predict_dataframe(row)}

    @app.post("/predict/batch")
    def predict_batch(batch_input: batch_input_model) -> dict[str, Any]:
        rows = [item.model_dump() for item in batch_input.inputs]
        df = pd.DataFrame(rows, columns=input_names)
        return {"predictions": _predict_dataframe(df)}

    @app.post("/invocations")
    def invocations(payload: input_model) -> dict[str, Any]:
        """MLflow-compatible scoring endpoint."""
        row = pd.DataFrame([payload.model_dump()], columns=input_names)
        return {"predictions": _predict_dataframe(row)}

    @app.get("/test")
    def test_prediction() -> dict[str, Any]:
        if example_data is None:
            raise HTTPException(
                status_code=404,
                detail="No saved input example is available for this model.",
            )
        df_predict = pd.DataFrame(
            example_data["data"],
            columns=example_data["columns"],
        )
        return {"predictions": _predict_dataframe(df_predict)}

    return app
