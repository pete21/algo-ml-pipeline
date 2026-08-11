# import matplotlib.pyplot as plt
# import seaborn as sns

import logging
import os

import dvc.api
import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from src.backtesting.optimization import objective
from src.data_utils.utils import get_dates
from src.model.mlflow_utils import (
    find_latest_experiment,
    load_model_params_from_experiment,
    search_positive_value_runs,
)
from src.model.model_building import load_data

EXPERIMENT_ID = 109
NUM_BEST_TRIALS = 10
NUM_EVALUATIONS_PER_TRIAL = 10

load_dotenv()

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def main():

    # Load parameters from the root directory
    params = dvc.api.params_show('params.yaml')
    
    model_evaluation_params = params['model_evaluation']
    experiment_id = model_evaluation_params['experiment_id']
    num_best_trials = model_evaluation_params['num_best_trials']
    num_evaluations_per_trial = model_evaluation_params['num_evaluations_per_trial']

    model_building_params = params['model_building']

    # Load the preprocessed data from the interim directory
    data = load_data(data_path=model_building_params['data_path'], params=model_building_params)
    
    cutoff_date = data[model_building_params['index_base']].index.date.min()+pd.Timedelta(30, "D")
    for d in data:
        data[d] = data[d].loc[data[d].index.date>=cutoff_date-pd.Timedelta(21, "D")]
    _, unique_weekdates = get_dates(data, model_building_params['index_base'])

    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    client = MlflowClient()

    experiment = find_latest_experiment(client, {"project_name": model_building_params['project_name'], "stage": "building"}, experiment_id=experiment_id)
    experiment_id = experiment.experiment_id
    print(f"Experiment: {experiment}")


    positive_value_run_params = search_positive_value_runs(experiment, num_runs=num_best_trials)

    for positive_value_run_param in positive_value_run_params:
        run_name = positive_value_run_param['tags.mlflow.runName']
        print(f"Run name: {run_name}")
        model_params = load_model_params_from_experiment(experiment, logger=logger, run_name=run_name)
        print(f"Loaded model params: {model_params}")

        params['run_name'] = run_name
        for i in range(num_evaluations_per_trial):
            print(f"Evaluation {i+1} of {NUM_EVALUATIONS_PER_TRIAL} for run {run_name}")
            optimisation_score = objective(None, data, model_building_params, cutoff_date, unique_weekdates, experiment_id, model_params_override=model_params)
            print(f"Optimisation score: {optimisation_score}")

if __name__ == '__main__':
    main()
