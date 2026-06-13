import os
import json
import mlflow
import logging
import pandas as pd
from mlflow.tracking import MlflowClient

from dotenv import load_dotenv

load_dotenv()

def create_mlflow_experiment(experiment_name: str, mlflow_tracking_uri: str, tags: dict = {}, logger: logging.Logger = None)->str:
    """
    Function to create an MLFlow experiment with a defined experiment name.
    """

    # Set MLFLow tracking URI
    print("Setting MLFlow tracking URI...")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logger.info("MLFlow Tracking URI URI set as: %s", mlflow_tracking_uri)

    logger.info("Creating experiment...")
    print("Creating experiment...")

    try:
        # Create the experiment. It returns the ID of the created experiment.
        experiment_id = mlflow.create_experiment(name=experiment_name, tags=tags)
        logger.info(f"Experiment '{experiment_name}' created with ID: {experiment_id}")
    except mlflow.exceptions.MlflowException as e:
        # Handle cases where the experiment might already exist
        logger.debug(f"Experiment '{experiment_name}' already exists.")
        # Optionally, get the ID of the existing experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing experiment ID: {experiment_id}")
        return experiment_id
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        raise
    return experiment_id


def find_latest_experiment(client: MlflowClient, tags: dict):
    """Find the most recent MLflow experiment tagged for the building stage."""
    filter_string = " AND ".join(
        f"tags.{key} = '{value}'" for key, value in tags.items()
    )
    experiments = client.search_experiments(filter_string=filter_string)
    if not experiments:
        raise ValueError(
            f"No MLflow experiments found with tags: {tags}"
        )
    return max(experiments, key=lambda exp: exp.last_update_time or exp.creation_time)


def fetch_trial_runs_dataframe(experiment_id: str) -> pd.DataFrame:
    """Fetch metrics and parameters for all non-nested Trial_* runs in an experiment."""
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="run_name LIKE 'Trial_%'",
    )
    parent_run_id_col = "tags.mlflow.parentRunId"
    if parent_run_id_col in runs_df.columns:
        runs_df = runs_df[runs_df[parent_run_id_col].isna()]
    return runs_df


def load_model_params_from_experiment(client: MlflowClient, tags: dict, logger: logging.Logger) -> dict:
    """Load model_params artifact from the best run of the latest experiment."""
    experiment = find_latest_experiment(client, tags)
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


def save_model_params(model_params: dict, file_path: str, logger: logging.Logger = None) -> None:
    """Save the trained model parameters to a file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(model_params, file)
        if logger:
            logger.debug('Model parameters saved to %s', file_path)
        else:
            print('Model parameters saved to %s', file_path)
    except Exception as e:
        if logger:
            logger.error('Error occurred while saving the model parameters: %s', e)
        raise