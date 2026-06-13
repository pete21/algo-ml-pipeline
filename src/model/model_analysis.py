import numpy as np
import pandas as pd
import logging
import mlflow
import os
import json
from datetime import date, datetime
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from src.data_utils.utils import load_params
from src.model.mlflow_utils import (
    fetch_trial_runs_dataframe,
    find_latest_experiment,
    load_model_params_from_experiment,
)
from dotenv import load_dotenv

load_dotenv()

# logging configuration
logger = logging.getLogger('model_analysis')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_analysis_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)



def main():
    print("Starting model analysis process...")
    # Get root directory and resolve the path for params.yaml
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')

    # Load parameters from the root directory
    params = load_params(os.path.join(root_dir, 'params.yaml'), logger=logger)


    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    client = MlflowClient()
    model_params = load_model_params_from_experiment(client, json.loads(os.getenv('BUILDING_EXPERIMENT_TAGS')), logger=logger)
    print(f"Loaded model params: {model_params}")

    experiment = find_latest_experiment(client, json.loads(os.getenv('BUILDING_EXPERIMENT_TAGS')))
    experiment_id = experiment.experiment_id

    print(f"Experiment ID: {experiment_id}")

    trial_runs_df = fetch_trial_runs_dataframe(experiment_id)
    print(f"Fetched {len(trial_runs_df)} Trial_* runs (excluding nested runs)")
    print(trial_runs_df.head())


if __name__ == '__main__':
    main()
