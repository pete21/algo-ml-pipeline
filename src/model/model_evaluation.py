import dvc.api
import numpy as np
import pandas as pd
import logging
import mlflow
import os
# import matplotlib.pyplot as plt
# import seaborn as sns
import json
from datetime import date
from mlflow.tracking import MlflowClient

from src.backtesting.optimization import objective
from src.data_utils.utils import get_dates
from src.model.mlflow_utils import find_latest_experiment, load_model_params_from_experiment, save_model_params
from src.model.model_building import load_data
from dotenv import load_dotenv

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

    # Load parameters from the root directory
    params = dvc.api.params_show('params.yaml')['model_building']

    # Load the preprocessed data from the interim directory
    data = load_data(data_path=params['data_path'], params=params)
    
    cutoff_date = date(2021,8,1)
    for d in data:
        data[d] = data[d].loc[data[d].index.date>=cutoff_date-pd.Timedelta(21, "D")]
    _, unique_weekdates = get_dates(data, params['index_base'])

    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    client = MlflowClient()

    experiment = find_latest_experiment(client, json.loads(os.getenv('BUILDING_EXPERIMENT_TAGS')))
    experiment_id = experiment.experiment_id
    print(f"Experiment ID: {experiment_id}")

    model_params = load_model_params_from_experiment(experiment, logger=logger)
    print(f"Loaded model params: {model_params}")

    optimisation_score = objective(None, data, params, cutoff_date, unique_weekdates, experiment_id, model_params_override=model_params)
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

    # Save the trained model in the root directory
    # print("Saving model parameters to json...")
    model_params_path = os.path.join(params['models_path'], f"model_params_evaluation_{run_id}.json")
    save_model_params(model_params=model_params, file_path=model_params_path, logger=logger)

    with mlflow.start_run(run_id=run_id) as run:
        mlflow.log_metric('optimisation_score', optimisation_score)
        mlflow.log_artifact(local_path=model_params_path, artifact_path='model_params')


if __name__ == '__main__':
    main()
