## Endpoints

Endpoint	Description

GET /docs   Swagger UI (request schema from signature)

GET /openapi.json   OpenAPI spec

GET /model/signature    Raw MLflow input/output signature

POST /predict   Single-row prediction

POST /predict/batch Batch prediction

POST /invocations   MLflow-compatible scoring endpoint

GET /test   Predict using the saved input example artifact

GET /health Health check


### Run the server

```bash
/home/piotr/venv/algo-ml-pipeline/bin/python -m src.serving.model_run
```

### Optional env vars

- `REGISTERED_MODEL_NAME` (default: `{MODEL_NAME}_20260525`)
- `MODEL_VERSION_ALIAS` (default: `Staging`)
- `MODEL_SERVE_PORT` (default: `8000`)

### Docker

Build & Run with the bundled `src/serving/.env`:

```bash
cd src/serving/
cp .env.example .env
docker build -f ./Dockerfile -t algo-ml-model-serving .
docker run --rm \
  --env-file ./.env \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000/ \
  --add-host=host.docker.internal:host-gateway \
  -p 8000:8000 \
  algo-ml-model-serving
```

Or use docker compose from `src/serving`:

```bash
cd src/serving/
docker compose up --build
```

Swagger UI: `http://localhost:{MODEL_SERVE_PORT}/docs`

When running in Docker, set `MLFLOW_TRACKING_URI` to a host reachable from the container
(for example `http://host.docker.internal:5000/` or `http://mlflow:5000/` instead of `http://localhost:5000/`).

