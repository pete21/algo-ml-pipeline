# import matplotlib.pyplot as plt
# import seaborn as sns
import json
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
)
from src.model.model_building import load_data

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
    params = dvc.api.params_show('params.yaml')['model_building']

    # Load the preprocessed data from the interim directory
    data = load_data(data_path=params['data_path'], params=params)
    
    cutoff_date = data[params['index_base']].index.date.min()+pd.Timedelta(30, "D")
    for d in data:
        data[d] = data[d].loc[data[d].index.date>=cutoff_date-pd.Timedelta(21, "D")]
    _, unique_weekdates = get_dates(data, params['index_base'])

    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    client = MlflowClient()

    experiment = find_latest_experiment(client, {"project_name": params['project_name'], "stage": "building"}, experiment_id=109)
    experiment_id = experiment.experiment_id
    print(f"Experiment ID: {experiment_id}")

    model_params = load_model_params_from_experiment(experiment, logger=logger, run_name='Trial_664')
    print(f"Loaded model params: {model_params}")

    optimisation_score = objective(None, data, params, cutoff_date, unique_weekdates, experiment_id, model_params_override=model_params)
    print(f"Optimisation score: {optimisation_score}")


    # print("Searching for run_id with run name: Evaluation")
    # # get run_id with run name 
    # run_object = mlflow.search_runs(filter_string="run_name = 'Evaluation'")
    # run_id = run_object["run_id"][0]

    # tags = {
    #     "project_name": "xgb-dax-pipeline",
    #     "stage": "evaluation",
    #     "mlflow.project_name.content": params['model_evaluation']['project_name'],
    # }
    # print("Setting tags for experiment: ", tags)
    # mlflow.set_experiment_tags(tags)

    # Save the trained model in the root directory
    # print("Saving model parameters to json...")
    # model_params_path = os.path.join(params['models_path'], f"model_params_evaluation_{run_id}.json")
    # save_model_params(model_params=model_params, file_path=model_params_path, logger=logger)

    # with mlflow.start_run(run_id=run_id) as run:
    #     mlflow.log_artifact(local_path=model_params_path, artifact_path='model_params')


if __name__ == '__main__':
    main()
