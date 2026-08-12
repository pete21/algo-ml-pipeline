import json
import logging
import os

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow.entities import Experiment
from mlflow.tracking import MlflowClient

load_dotenv()

def create_mlflow_experiment(experiment_name: str, mlflow_tracking_uri: str, tags: dict = {}, logger: logging.Logger | None = None)->str:
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


def find_latest_experiment(client: MlflowClient, tags: dict, experiment_id: int | None = None):
    """Find the most recent MLflow experiment tagged for the building stage."""
    filter_string = " AND ".join(
        f"tags.{key} = '{value}'" for key, value in tags.items()
    )
    experiments = client.search_experiments(filter_string=filter_string) if experiment_id is None else [client.get_experiment(experiment_id)]
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


def load_model_params_from_experiment(experiment: Experiment, logger: logging.Logger, run_name: str | None = None) -> dict:
    """Load model_params artifact from the best run of the latest experiment."""
    if run_name:
        best_run_name = run_name
    else:
        best_run_name = experiment.tags.get("best_run_name")
        if not best_run_name:
            raise ValueError(
                f"Experiment '{experiment.name}' (id={experiment.experiment_id}) "
                "is missing the 'best_run_name' tag"
            )

    print(
        f"Using experiment '{experiment.name}' (id={experiment.experiment_id}), "
        f"run name='{best_run_name}'"
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


# Search positive value runs and return run params
def search_positive_value_runs(experiment: Experiment, num_runs: int = 10) -> list[dict]:
    """Search all positive value runs and return the run params of them."""
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.optimisation_score > 0",
    )
    print(f"Found {len(runs)} positive value runs")
    if runs.empty:
        return []

    # Sort rows by metrics.optimisation_score in descending order
    runs = runs.sort_values(by="metrics.optimisation_score", ascending=False)
    # Extract columns with "params" prefix
    params_columns = [col for col in runs.columns if col.startswith("params.")]
    runs = runs[params_columns + ["metrics.optimisation_score", "metrics.win_rate", "metrics.total_trades", "metrics.sharpe_ratio", "metrics.total_profit", "tags.mlflow.runName"]]
    # print(runs.to_dict(orient="records"))
    return runs.iloc[:num_runs].to_dict(orient="records")


def save_model_params(model_params: dict, file_path: str, logger: logging.Logger | None = None) -> None:
    """Save the trained model parameters to a file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(model_params, file)
        if logger:
            logger.debug('Model parameters saved to %s', file_path)
        else:
            print(f'Model parameters saved to {file_path}')
    except Exception as e:
        if logger:
            logger.error('Error occurred while saving the model parameters: %s', e)
        raise

def find_model_versions(client: MlflowClient, model_name: str) -> list[int]:
    """Find the model versions for a given model name."""
    model_versions = client.search_model_versions(f"name='{model_name}'")
    return [int(version.version) for version in model_versions]

def find_latest_model_version(client: MlflowClient, model_name: str) -> int:
    """Find the latest model version for a given model name."""
    model_versions = find_model_versions(client, model_name)
    max_version = max(model_versions)
    print(f"Model version: {max_version}")
    # print(f"Model version creation time: {max_version.creation_timestamp}")
    # print(f"Model version description: {max_version.description}")
    # print(f"Model version source: {max_version.source}")
    # print(f"Model version status: {max_version.status}")
    # print(f"Model version run_id: {max_version.run_id}")
    # print(f"Model version tags: {max_version.tags}")
    # print(f"Model version aliases: {max_version.aliases}")
    return max_version

def set_alias_to_model_version(client: MlflowClient, model_name: str, version: int, alias: str, logger: logging.Logger | None = None) -> None:
    """Set an alias to a model version."""
    try:
        if logger:
            logger.info(f"Setting alias {alias} to model version {version} for model {model_name}")
        else:
            print(f"Setting alias {alias} to model version {version} for model {model_name}")
        client.set_registered_model_alias(name=model_name, version=str(version), alias=alias)
    except Exception as e:
        logger.error(f"Error setting alias to model version: {e}")
        raise
