import logging
import os
from datetime import datetime

import dvc.api
import mlflow
import optuna
import pandas as pd
import pytz
from dotenv import load_dotenv

from src.backtesting.optimization import objective
from src.data_utils.utils import get_dates
from src.model.mlflow_utils import create_mlflow_experiment

load_dotenv()


# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def ceiling_division(n, d):
    return -(n // -d)

def load_data(data_path: str, params: dict) -> dict:
    """Load data from a CSV file."""
    local_timezone = pytz.timezone(params['local_timezone'])

    try:
        data = {}
        print(os.path.join(data_path, params['file_name'].format(timeframe=params['timeframes'][params['index_base']])))
        print("Loading data for index base: ", params['index_base'])
        data[params['index_base']] = pd.read_csv(os.path.join(data_path, params['file_name'].format(timeframe=params['timeframes'][params['index_base']])), parse_dates=True, index_col='date')
        # data[params['data_preprocessing']['index_base']]["high_time"] = pd.to_datetime(data[params['data_preprocessing']['index_base']]["high_time"])
        # data[params['data_preprocessing']['index_base']]["low_time"] = pd.to_datetime(data[params['data_preprocessing']['index_base']]["low_time"])
        data[params['index_base']]['local_date'] = data[params['index_base']].index.tz_localize('UTC').tz_convert(local_timezone)
        data[params['index_base']]["date_merge"] = data[params['index_base']].index
        print(data[params['index_base']].head())

        for i in params['indexes_higher']:
            print(os.path.join(data_path, params['file_name'].format(timeframe=params['timeframes'][i])))
            print("Loading data for index: ", i)
            data[i] = pd.read_csv(os.path.join(data_path, params['file_name'].format(timeframe=params['timeframes'][i])), parse_dates=True, index_col='date')
            # data[i]['local_date'] = data[i].index.tz_localize('UTC').tz_convert(local_timezone)
            data[i]["date_merge"] = (
                data[i].index
                + pd.to_timedelta(params['timeframe_minutes'][i], "m")
                - pd.to_timedelta(params['timeframe_minutes'][params['index_base']], "m")
            )

            # for each value of data[i]["date_merge"], if minute value of data[i]["date_merge"] is not divisible by params['timeframe_minutes'][params['index_base']], set the minute value to the next divisible value greater than the current value
            data[i]["date_merge"] = data[i]["date_merge"].apply(lambda x: x.replace(minute=ceiling_division(x.minute, params['timeframe_minutes'][params['index_base']]) * params['timeframe_minutes'][params['index_base']]))

            print(data[i].head())

        logger.debug('Data loaded from %s', data_path)
        return data
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


# def apply_dynamic_features(data: dict, params: dict, scaler: float) -> dict:
#     """Apply dynamic features to the data."""
#     try:
        
#         for i in params['data_ingestion']['indexes_higher']:
#             data[i] = dynamic_features(data[i], parameters, params['data_ingestion']['timeframe_scalers'][i], col_close="Close", col_high="High", col_low="Low")
#         data[params['data_ingestion']['index_base']] = dynamic_features(data[params['data_ingestion']['index_base']], parameters, params['data_ingestion']['timeframe_scalers'][params['data_ingestion']['index_base']], col_close="Close", col_high="High", col_low="Low")
#         return data
#     except Exception as e:
#         logger.error('Error during dynamic features transformation: %s', e)
#         raise


# def train_lgbm(X_train: np.ndarray, y_train: np.ndarray, learning_rate: float, max_depth: int, n_estimators: int) -> lgb.LGBMClassifier:
#     """Train a LightGBM model."""
#     try:
#         best_model = lgb.LGBMClassifier(
#             objective='multiclass',
#             num_class=3,
#             metric="multi_logloss",
#             is_unbalance=True,
#             class_weight="balanced",
#             reg_alpha=0.1,  # L1 regularization
#             reg_lambda=0.1,  # L2 regularization
#             learning_rate=learning_rate,
#             max_depth=max_depth,
#             n_estimators=n_estimators
#         )
#         best_model.fit(X_train, y_train)
#         logger.debug('LightGBM model training completed')
#         return best_model
#     except Exception as e:
#         logger.error('Error during LightGBM model training: %s', e)
#         raise

# def train_XGBoost(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> xgb.XGBClassifier:
#     """Train an XGBoost model."""
#     try:
#         best_model = xgb.XGBClassifier(
#             objective='multiclass',
#             num_class=3,
#             metric="multi_logloss",
#             is_unbalance=True,
#             class_weight="balanced",
#         )
#         return best_model
#     except Exception as e:
#         logger.error('Error during XGBoost model training: %s', e)
#         raise



# def get_root_directory() -> str:
#     """Get the root directory (two levels up from this script's location)."""
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     return os.path.abspath(os.path.join(current_dir, '../../'))


# def get_or_create_experiment(experiment_name):
#   """
#   Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

#   This function checks if an experiment with the given name exists within MLflow.
#   If it does, the function returns its ID. If not, it creates a new experiment
#   with the provided name and returns its ID.

#   Parameters:
#   - experiment_name (str): Name of the MLflow experiment.

#   Returns:
#   - str: ID of the existing or newly created MLflow experiment.
#   """

#   if experiment := mlflow.get_experiment_by_name(experiment_name):
#       return experiment.experiment_id
#   else:
#       return mlflow.create_experiment(experiment_name)



def main():
    print("Starting model building process...")
    try:
        # Get root directory and resolve the path for params.yaml
        # root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')

        # Load parameters from the root directory
        # params = load_params(os.path.join(root_dir, 'params.yaml'), logger=logger)
        # Load parameters from the params.yaml in the root directory
        params = dvc.api.params_show('params.yaml')['model_building']
        print(f"Params: {params}")
        # model_params = load_json_params(os.path.join(root_dir, 'model_params.json'), logger=logger)

        # Load the preprocessed data from the interim directory
        data = load_data(data_path=params['data_path'], params=params)
        
        unique_dates, unique_weekdates = get_dates(data, params['index_base'])
        cutoff_date = data[params['index_base']].index.date.min()+pd.Timedelta(21, "D")

        experiment_name = f'{params["project_name"]}-v{params["version"]}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        if params['evals_strategy']:
            experiment_name += '-evals'
        else:
            experiment_name += '-trading'
        print(f"Experiment name: {experiment_name}")

        experiment_id = create_mlflow_experiment(experiment_name=experiment_name, mlflow_tracking_uri=os.getenv('MLFLOW_TRACKING_URI'), tags={}, logger=logger)
        mlflow.set_experiment(experiment_id=experiment_id)
        # experiment = mlflow.get_experiment(experiment_id=experiment_id)

        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, data, params, cutoff_date, unique_weekdates, experiment_id), n_trials=params['n_trials'])
        print("Best trial number: ", study.best_trial.number)
        print("Best parameters: ", study.best_params)
        print("Best score:", study.best_value)


        print("Saving study trials to csv...")
        optuna_trials_path = os.path.join(params['models_path'],f"{experiment_name}_trials.csv")
        study_trials = study.trials_dataframe().sort_values(by=['value'], ascending=False)
        print("Study trials: ", study_trials.head())
        study_trials.to_csv(optuna_trials_path, index=False)
        study_trials.to_csv(os.path.join(params['models_path'],"optuna_trials.csv"), index=False)

        print(f"Searching for run_id with run name: Trial_{study.best_trial.number}")
        # get run_id with run name 
        run_object = mlflow.search_runs(filter_string=f"run_name = 'Trial_{study.best_trial.number}'")
        run_id = run_object["run_id"][0]

        with mlflow.start_run(run_id=run_id) as run:
            mlflow.log_metric('best', True)

        tags = {
            "project_name": params['project_name'],
            "stage": "building",
            "mlflow.note.content": f"Project name: {params['project_name']}",
            "optimizer": "optuna",
            "model_family": params['model_type'],
            "model_name": params['model_type'] + '_v' + str(params['version']),
            "best_trial_number": study.best_trial.number,
            "best_run_name": f"Trial_{study.best_trial.number}",
            "best_run_id": run_id,
        }
        print("Setting tags for experiment: ", tags)
        mlflow.set_experiment_tags(tags)

        print(f"Searching for last run_id with run name: Trial_{len(study.trials)-1}")
        # get run_id with run name 
        run_object = mlflow.search_runs(filter_string=f"run_name = 'Trial_{len(study.trials)-1}'")
        run_id = run_object["run_id"][0]
        
        with mlflow.start_run(run_id=run_id) as run:
            mlflow.log_artifact(local_path=os.path.join(params['models_path'],"optuna_trials.csv"), artifact_path='optuna_trials')

    except Exception as e:
        logger.error('Failed to complete the feature engineering and model building process: %s', e)
        print(f"Error: {e}")
    print("Model building process completed successfully.")


if __name__ == '__main__':
    main()
