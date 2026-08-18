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

POST /load_model

Request body:
```json
{
  "registered_model_name": "my_model",
  "model_version_alias": "Staging"
}
```

On success it returns status: ok plus the updated serving config (model URI, version, aliases, params, etc.). On failure it returns 400 for validation/load issues or 500 for unexpected errors, and the previous model stays in place.

```bash
curl -X POST http://localhost:8100/load_model \
  -H "Content-Type: application/json" \
  -d '{"registered_model_name": "my_model", "model_version_alias": "Production"}'
```


## How it works

model_run.py bootstraps via ServingState.from_registered_model(...) instead of passing individual model objects into build_serving_app.
ServingState holds the active model, metadata, params, signature models, and input feature names.
load_model() loads from MLflow via the existing load_registered_model() helper, then atomically replaces all serving state under a lock.
All prediction endpoints (/predict, /predict/batch, /invocations, /test) and config endpoints (/, /model/params, /model/signature) read from ServingState, so they use the newly loaded model immediately.
App title/version are updated after a successful reload.



### Run the server

```bash
/home/piotr/venv/algo-ml-pipeline/bin/python -m src.serving.model_run
```

### Optional env vars

- `PORT` (default: `8100`)

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
  -p 8100:8100 \
  algo-ml-model-serving
```

Or use docker compose from `src/serving`:

```bash
cd src/serving/
docker compose up --build
```

Swagger UI: `http://localhost:{PORT}/docs`

When running in Docker, set `MLFLOW_TRACKING_URI` to a host reachable from the container
(for example `http://host.docker.internal:5000/` or `http://mlflow:5000/` instead of `http://localhost:5000/`).

