import numpy as np
import pandas as pd
import logging
import mlflow
import os
# import matplotlib.pyplot as plt
# import seaborn as sns
import json
from datetime import date
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from src.backtesting.optimization import objective
from src.data_utils.utils import load_params, get_dates
from src.model.model_building import load_data

os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://192.168.10.250:9900"

MLFLOW_TRACKING_URI = "http://localhost:5000/"
BUILDING_EXPERIMENT_TAGS = {
    "project_name": "xgb-dax-pipeline",
    "stage": "building",
}

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)



# def load_model(model_path: str):
#     """Load the trained model."""
#     try:
#         with open(model_path, 'rb') as file:
#             model = pickle.load(file)
#         logger.debug('Model loaded from %s', model_path)
#         return model
#     except Exception as e:
#         logger.error('Error loading model from %s: %s', model_path, e)
#         raise


# def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
#     """Load the saved TF-IDF vectorizer."""
#     try:
#         with open(vectorizer_path, 'rb') as file:
#             vectorizer = pickle.load(file)
#         logger.debug('TF-IDF vectorizer loaded from %s', vectorizer_path)
#         return vectorizer
#     except Exception as e:
#         logger.error('Error loading vectorizer from %s: %s', vectorizer_path, e)
#         raise


def find_latest_building_experiment(client: MlflowClient):
    """Find the most recent MLflow experiment tagged for the building stage."""
    filter_string = " AND ".join(
        f"tags.{key} = '{value}'" for key, value in BUILDING_EXPERIMENT_TAGS.items()
    )
    experiments = client.search_experiments(filter_string=filter_string)
    if not experiments:
        raise ValueError(
            f"No MLflow experiments found with tags: {BUILDING_EXPERIMENT_TAGS}"
        )
    return max(experiments, key=lambda exp: exp.last_update_time or exp.creation_time)


def load_model_params_from_building_experiment(client: MlflowClient) -> dict:
    """Load model_params artifact from the best run of the latest building experiment."""
    experiment = find_latest_building_experiment(client)
    best_run_name = experiment.tags.get("best_run_name")
    if not best_run_name:
        raise ValueError(
            f"Experiment '{experiment.name}' (id={experiment.experiment_id}) "
            "is missing the 'best_run_name' tag"
        )

    print(
        f"Using experiment '{experiment.name}' (id={experiment.experiment_id}), "
        f"best_run_name='{best_run_name}'"
    )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"run_name = '{best_run_name}'",
    )
    if runs.empty:
        raise ValueError(
            f"No run named '{best_run_name}' found in experiment "
            f"'{experiment.name}' (id={experiment.experiment_id})"
        )

    run_id = runs.iloc[0]["run_id"]
    artifact_dir = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="model_params",
    )

    json_files = [
        os.path.join(artifact_dir, filename)
        for filename in os.listdir(artifact_dir)
        if filename.endswith(".json")
    ]
    if not json_files:
        raise ValueError(
            f"No JSON model_params artifact found for run '{best_run_name}' "
            f"(run_id={run_id})"
        )

    with open(json_files[0], "r") as file:
        model_params = json.load(file)

    logger.debug(
        "Model params loaded from MLflow run '%s' (run_id=%s)",
        best_run_name,
        run_id,
    )
    return model_params


# def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
#     """Save the model run ID and path to a JSON file."""
#     try:
#         # Create a dictionary with the info you want to save
#         model_info = {
#             'run_id': run_id,
#             'model_path': model_path
#         }
#         # Save the dictionary as a JSON file
#         with open(file_path, 'w') as file:
#             json.dump(model_info, file, indent=4)
#         logger.debug('Model info saved to %s', file_path)
#     except Exception as e:
#         logger.error('Error occurred while saving the model info: %s', e)
#         raise


def main():

    # Get root directory and resolve the path for params.yaml
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')

    # Load parameters from the root directory
    params = load_params(os.path.join(root_dir, 'params.yaml'), logger=logger)

    # Load the preprocessed data from the interim directory
    data = load_data(data_path=params['model_evaluation']['data_path'], params=params)
    
    cutoff_date = date(2025,1,2)
    for d in data:
        data[d] = data[d].loc[data[d].index.date>=cutoff_date-pd.Timedelta(21, "D")]
    unique_dates, unique_weekdates, mondays_indexes = get_dates(data, params['model_building']['index_base'])
    print(mondays_indexes)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    model_params = load_model_params_from_building_experiment(client)
    print(f"Loaded model params: {model_params}")

    experiment = find_latest_building_experiment(client)
    experiment_id = experiment.experiment_id
    optimisation_score = objective(None, data, params['model_evaluation'], cutoff_date, unique_dates, mondays_indexes, experiment_id, model_params_override=model_params)
    print(f"Optimisation score: {optimisation_score}")


    print(f"Searching for run_id with run name: Evaluation")
    # get run_id with run name 
    run_object = mlflow.search_runs(filter_string=f"run_name = 'Evaluation'")
    run_id = run_object["run_id"][0]

    # tags = {
    #     "project_name": "xgb-dax-pipeline",
    #     "stage": "evaluation",
    #     "mlflow.note.content": params['model_evaluation']['note'],
    # }
    # print("Setting tags for experiment: ", tags)
    # mlflow.set_experiment_tags(tags)

    with mlflow.start_run(run_id=run_id) as run:
        mlflow.log_metric('optimisation_score', optimisation_score)


if __name__ == '__main__':
    main()
