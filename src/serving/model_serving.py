import json
import os
import threading
from typing import Any, Optional, Type

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from mlflow.models import get_model_info
from pydantic import BaseModel, Field, ValidationError, create_model
from xgboost import DMatrix


class TabularPredictRequest(BaseModel):
    columns: list[str]
    data: list[list[Any]]


class LoadModelRequest(BaseModel):
    registered_model_name: str
    model_version_alias: str


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


def _build_signature_models(
    model_info: Any,
) -> tuple[Type[BaseModel], Type[BaseModel], list[str]]:
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
    return input_model, batch_input_model, input_names


def load_registered_model(
    tracking_uri: str,
    registered_model_name: str,
    model_version_alias: str,
) -> tuple[Any, Any, str, Any, dict, Optional[dict]]:
    """Load a registered MLflow XGBoost model and its metadata."""
    client = MlflowClient(tracking_uri=tracking_uri)
    try:
        model_version_info = client.get_model_version_by_alias(
            registered_model_name,
            model_version_alias,
        )
    except Exception as e:
        print(f"Error getting model version by alias: {e}")
        return None, None, None, None, {}, {}
        # raise ValueError(f"Error getting model version by alias: {e}")

    model_uri = model_version_info.source
    model_info = get_model_info(model_uri)
    model = mlflow.xgboost.load_model(model_uri)

    model_params = {}
    artifact_dir = mlflow.artifacts.download_artifacts(
        run_id=model_version_info.run_id,
        artifact_path="model_params",
    )
    json_files = [
        os.path.join(artifact_dir, filename)
        for filename in os.listdir(artifact_dir)
        if filename.endswith(".json")
    ]
    if not json_files:
        raise ValueError(
            f"No JSON model_params artifact found for run_id={model_version_info.run_id}"
        )
    with open(json_files[0], "r", encoding="utf-8") as file:
        model_params = json.load(file)

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

    return model, model_info, model_uri, model_version_info, model_params, example_data


class ServingState:
    """Mutable serving configuration that can be updated at runtime."""

    def __init__(self, tracking_uri: str):
        self.tracking_uri = tracking_uri
        self._lock = threading.Lock()
        self.model: Any = None
        self.model_info: Any = None
        self.model_uri: str = ""
        self.registered_model_name: str = ""
        self.model_version_info: Any = None
        self.model_params: dict = {}
        self.example_data: Optional[dict] = None
        self.input_names: list[str] = []
        self.input_model: Optional[Type[BaseModel]] = None
        self.batch_input_model: Optional[Type[BaseModel]] = None

    @classmethod
    def from_registered_model(
        cls,
        tracking_uri: str,
        registered_model_name: str,
        model_version_alias: str,
    ) -> "ServingState":
        state = cls(tracking_uri)
        state.load_model(registered_model_name, model_version_alias)
        return state

    def _set_loaded_model(
        self,
        registered_model_name: str,
        model: Any,
        model_info: Any,
        model_uri: str,
        model_version_info: Any,
        model_params: dict,
        example_data: Optional[dict],
    ) -> None:
        if model_info is not None:
            input_model, batch_input_model, input_names = _build_signature_models(model_info)
        else:
            input_model = None
            batch_input_model = None
            input_names = []
        self.model = model
        self.model_info = model_info
        self.model_uri = model_uri
        self.registered_model_name = registered_model_name
        self.model_version_info = model_version_info
        self.model_params = model_params
        self.example_data = example_data
        self.input_names = input_names
        self.input_model = input_model
        self.batch_input_model = batch_input_model

    def load_model(self, registered_model_name: str, model_version_alias: str) -> dict[str, Any]:
        """Load a registered model and replace the active serving configuration."""
        loaded = load_registered_model(
            self.tracking_uri,
            registered_model_name,
            model_version_alias,
        )
        with self._lock:
            self._set_loaded_model(registered_model_name, *loaded)
            return self.as_config()

    def as_config(self) -> dict[str, Any]:
        if self.model_info is None:
            return {
                "message": "MLflow model prediction API - missing model info",
            }
        return {
            "message": "MLflow model prediction API",
            "model_uri": self.model_uri,
            "registered_model_name": self.registered_model_name,
            "model_version": self.model_version_info.version,
            "model_version_alias": list(self.model_version_info.aliases),
            "input_features": len(self.input_names),
            "docs_url": "/docs",
            "openapi_url": "/openapi.json",
            "model_params": self.model_params,
        }

    def predict_dataframe(self, df: pd.DataFrame) -> list[list[float]]:
        with self._lock:
            missing_columns = set(self.input_names) - set(df.columns)
            if missing_columns:
                raise HTTPException(
                    status_code=422,
                    detail=f"Missing input features: {sorted(missing_columns)}",
                )
            ordered_df = df[self.input_names]
            predictions = self.model.predict(DMatrix(ordered_df))
        return predictions.tolist()


def build_serving_app(state: ServingState) -> FastAPI:
    """Create a FastAPI app with Swagger docs derived from the MLflow input signature."""
    if state.model_info is None:
        app = FastAPI(title="MLflow Model Serving - missing model info")
    else:
        app = FastAPI(
            title=f"MLflow Model Serving: {state.registered_model_name}",
            description=(
                "Prediction API with request schema generated from "
                "`model_info.signature.inputs`."
            ),
            version=str(state.model_version_info.version),
        )

    @app.get("/")
    def root() -> dict[str, Any]:
        with state._lock:
            return state.as_config()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/model/params")
    def get_model_params() -> dict[str, Any]:
        with state._lock:
            if state.model_info is None:
                raise HTTPException(status_code=400, detail=state.as_config())
            return state.model_params

    @app.get("/model/signature")
    def model_signature() -> dict[str, Any]:
        with state._lock:
            if state.model_info is None:
                raise HTTPException(status_code=400, detail=state.as_config())
            return {
                "inputs": state.model_info.signature.inputs.to_dict(),
                "outputs": state.model_info.signature.outputs.to_dict()
                if state.model_info.signature.outputs
                else None,
            }

    @app.post("/load_model")
    def load_model(payload: LoadModelRequest) -> dict[str, Any]:
        """Load a registered MLflow model and update the active serving configuration."""
        try:
            config = state.load_model(
                payload.registered_model_name,
                payload.model_version_alias,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if state.model_info is not None:
            app.title = f"MLflow Model Serving: {state.registered_model_name}"
            app.version = str(state.model_version_info.version)
            return {"status": "ok", **config}
        else:
            app.title = "MLflow Model Serving - missing model info"
            return {"status": "error", **config}

    @app.post("/predict")
    def predict(payload: TabularPredictRequest) -> dict[str, Any]:
        if state.model_info is None:
            raise HTTPException(status_code=400, detail=state.as_config())
        df = pd.DataFrame(payload.data, columns=payload.columns)
        return {"predictions": state.predict_dataframe(df)}

    @app.post("/predict/batch")
    def predict_batch(payload: dict[str, Any]) -> dict[str, Any]:
        with state._lock:
            if state.model_info is None:
                raise HTTPException(status_code=400, detail=state.as_config())
            batch_input_model = state.batch_input_model
            input_names = state.input_names
        try:
            batch_input = batch_input_model.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        rows = [item.model_dump() for item in batch_input.inputs]
        df = pd.DataFrame(rows, columns=input_names)
        return {"predictions": state.predict_dataframe(df)}

    @app.post("/invocations")
    def invocations(payload: dict[str, Any]) -> dict[str, Any]:
        """MLflow-compatible scoring endpoint."""
        with state._lock:
            if state.model_info is None:
                raise HTTPException(status_code=400, detail=state.as_config())
            input_model = state.input_model
            input_names = state.input_names
        try:
            validated = input_model.model_validate(payload)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        row = pd.DataFrame([validated.model_dump()], columns=input_names)
        return {"predictions": state.predict_dataframe(row)}

    @app.get("/test")
    def test_prediction() -> dict[str, Any]:
        with state._lock:
            if state.model_info is None:
                raise HTTPException(status_code=400, detail=state.as_config())
            example_data = state.example_data
        if example_data is None:
            raise HTTPException(
                status_code=404,
                detail="No saved input example is available for this model.",
            )
        df_predict = pd.DataFrame(
            example_data["data"],
            columns=example_data["columns"],
        )
        return {"predictions": state.predict_dataframe(df_predict)}

    return app
