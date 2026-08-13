import logging
import os
from datetime import datetime

import dvc.api
import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from src.backtesting.optimization import train_register_model
from src.data_utils.utils import get_dates
from src.model.mlflow_utils import (
    find_latest_experiment,
    find_latest_model_version,
    load_model_params_from_experiment,
    set_alias_to_model_version,
)
from src.model.model_building import load_data

load_dotenv()

# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)



def main():

    # Load parameters from the root directory
    params = dvc.api.params_show('params.yaml')
    model_registration_params = params['model_registration']
    model_tag = model_registration_params['model_tag']
    experiment_id = model_registration_params['experiment_id']
    run_name = model_registration_params['run_name']
    model_building_params = params['model_building']

    # Load the preprocessed data from the interim directory
    data = load_data(data_path=model_building_params['data_path'], params=model_building_params)
    
    cutoff_date = datetime.today().date() - pd.Timedelta(days=364)
    for d in data:
        data[d] = data[d].loc[data[d].index.date>=cutoff_date]
    _, unique_weekdates = get_dates(data, model_building_params['index_base'])

    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    client = MlflowClient()

    experiment = find_latest_experiment(client, {"project_name": model_building_params['project_name'], "stage": "building"}, experiment_id=experiment_id)
    experiment_id = experiment.experiment_id
    print(f"Experiment: {experiment}")

    model_params = load_model_params_from_experiment(experiment, logger=logger, run_name=run_name)
    print(f"Loaded model params: {model_params}")

    last_weekdate = [d for d in unique_weekdates if d.weekday()==model_params['weekday']][-1]
    train_split_index = unique_weekdates.index(last_weekdate)
    print(f"Train split index: {train_split_index}")
    print(f"Train split date: {unique_weekdates[train_split_index]}")

    registered_model_name = train_register_model(data=data, params=model_building_params, unique_weekdates=unique_weekdates, train_split_index=train_split_index, experiment_id=experiment_id, model_params=model_params)
    print(f"Registered model name: {registered_model_name}")

    client = mlflow.MlflowClient()
    version = find_latest_model_version(client, registered_model_name)
    set_alias_to_model_version(client, registered_model_name, version, model_tag, logger=logger)
    print(f"Model version {version} set to alias {model_tag} for model {registered_model_name}")


if __name__ == '__main__':
    main()
