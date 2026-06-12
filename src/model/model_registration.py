import numpy as np
import pandas as pd
import logging
import mlflow
import os
import json
from datetime import date, datetime
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from src.backtesting.optimization import objective, train_register_model
from src.data_utils.utils import load_params, get_dates
from src.model.model_building import load_data
from src.model.model_evaluation import find_latest_building_experiment, load_model_params_from_building_experiment

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




def main():

    # Get root directory and resolve the path for params.yaml
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')

    # Load parameters from the root directory
    params = load_params(os.path.join(root_dir, 'params.yaml'), logger=logger)

    # Load the preprocessed data from the interim directory
    data = load_data(data_path=params['model_registration']['data_path'], params=params)
    
    cutoff_date = date.today() - pd.Timedelta(days=365)
    for d in data:
        data[d] = data[d].loc[data[d].index.date>=cutoff_date]
    unique_dates, unique_weekdates, mondays_indexes = get_dates(data, params['model_building']['index_base'])
    print(mondays_indexes)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    model_params = load_model_params_from_building_experiment(client)
    print(f"Loaded model params: {model_params}")

    model_params['num_splits'] = 1

    experiment = find_latest_building_experiment(client)
    experiment_id = experiment.experiment_id
    train_register_model(data, params['model_registration'], unique_dates, mondays_indexes[-1:], experiment_id, model_params)


if __name__ == '__main__':
    main()
