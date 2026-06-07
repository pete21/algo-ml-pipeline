
import mlflow
import numpy as np
import pandas as pd
import os
import json
import logging
from src.data_utils.utils import load_params, load_json_params, get_dates
from src.backtesting.optimization import objective
from datetime import date, datetime

import optuna
# from optuna.visualization import plot_param_importances, plot_contour, plot_slice
# from optuna.visualization import plot_contour
# from optuna.visualization import plot_edf
# from optuna.visualization import plot_intermediate_values
# from optuna.visualization import plot_optimization_history
# from optuna.visualization import plot_parallel_coordinate
# from optuna.visualization import plot_param_importances
# from optuna.visualization import plot_rank
# from optuna.visualization import plot_slice
# from optuna.visualization import plot_timeline
# from optuna.importance import get_param_importances

# Set credentials matching your .env file
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"

# Point to your remote MinIO instance (Notice the remote IP and port 9900)
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://192.168.10.250:9900"


# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(data_path: str, params: dict) -> dict:
    """Load data from a CSV file."""
    try:
        data = {}
        print(os.path.join(data_path, params['model_building']['file_name'].format(timeframe=params['model_building']['timeframes'][params['model_building']['index_base']])))
        print("Loading data for index base: ", params['model_building']['index_base'])
        data[params['model_building']['index_base']] = pd.read_csv(os.path.join(data_path, params['model_building']['file_name'].format(timeframe=params['model_building']['timeframes'][params['model_building']['index_base']])), parse_dates=True, index_col='date')
        # data[params['data_preprocessing']['index_base']]["high_time"] = pd.to_datetime(data[params['data_preprocessing']['index_base']]["high_time"])
        # data[params['data_preprocessing']['index_base']]["low_time"] = pd.to_datetime(data[params['data_preprocessing']['index_base']]["low_time"])
        
        for i in params['model_building']['indexes_higher']:
            print(os.path.join(data_path, params['model_building']['file_name'].format(timeframe=params['model_building']['timeframes'][i])))
            print("Loading data for index: ", i)
            data[i] = pd.read_csv(os.path.join(data_path, params['model_building']['file_name'].format(timeframe=params['model_building']['timeframes'][i])), parse_dates=True, index_col='date')
        
        data[params['model_building']['index_base']]["date_merge"] = data[params['model_building']['index_base']].index
        for i in params['model_building']['indexes_higher']:
            data[i]["date_merge"] = data[i].index
        #     data[i]["date_merge"] = pd.to_datetime(data[i]["date_merge"])

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


def save_model(model_params: dict, file_path: str) -> None:
    """Save the trained model parameters to a file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(model_params, file)
        logger.debug('Model parameters saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model parameters: %s', e)
        raise


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


def create_mlflow_experiment(experiment_name: str, mlflow_tracking_uri: str, tags: dict)->str:
    """
    Function to create an MLFlow experiment with a defined experiment name.
    """

    # Set MLFLow tracking URI
    print("Setting MLFlow tracking URI...")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logging.info("MLFlow Tracking URI URI set as: %s", mlflow_tracking_uri)

    logging.info("Creating experiment...")
    print("Creating experiment...")

    try:
        # Create the experiment. It returns the ID of the created experiment.
        experiment_id = mlflow.create_experiment(name=experiment_name)
        print(f"Experiment '{experiment_name}' created with ID: {experiment_id}")
    except mlflow.exceptions.MlflowException as e:
        # Handle cases where the experiment might already exist
        print(f"Experiment '{experiment_name}' already exists.")
        # Optionally, get the ID of the existing experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment ID: {experiment_id}")
        return experiment_id
    except Exception as e:
        print(f"Error creating experiment: {e}")
        raise
    return experiment_id


def main():
    try:
        # Get root directory and resolve the path for params.yaml
        root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')

        # Load parameters from the root directory
        params = load_params(os.path.join(root_dir, 'params.yaml'), logger=logger)
        # model_params = load_json_params(os.path.join(root_dir, 'model_params.json'), logger=logger)

        # Load the preprocessed data from the interim directory
        data = load_data(data_path=params['model_building']['data_path'], params=params)
        
        cutoff_date = date(2024,6,1)
        for d in data:
            data[d] = data[d].loc[data[d].index.date>=cutoff_date-pd.Timedelta(14, "D")]
        unique_dates, unique_weekdates, mondays_indexes = get_dates(data, params['model_building']['index_base'])
        print(mondays_indexes)
        splits_all = []

        experiment_name = f'xgb-dax-pipeline-runs-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        print(f"Experiment name: {experiment_name}")
        tags = {
            "project_name": "xgb-dax-pipeline",
            "mlflow.note.content": "This is the experiment for the xgb-dax-pipeline",
            "optimizer": "optuna",
            "model_family": "xgboost",
            "feature_set_version": 1,
        }

        experiment_id = create_mlflow_experiment(experiment_name=experiment_name, mlflow_tracking_uri="http://localhost:5000/", tags=tags)
        mlflow.set_experiment(experiment_id=experiment_id)
        # experiment = mlflow.get_experiment(experiment_id=experiment_id)

        #     try:

        with mlflow.start_run() as run:
            mlflow.log_params(params['model_building'])
            # mlflow.log_param('num_splits', num_splits)
            mlflow.log_param('start_time', datetime.now())
            mlflow.log_param('cutoff_date', cutoff_date)
            # mlflow.log_param('end_time', datetime.now())
            # mlflow.log_param('duration', datetime.now() - start_time)
            # mlflow.log_param('_strategy', stats[0]['_strategy'])
            # mlflow.log_metric('total_score', total_score)
            # mlflow.log_metric('scores_std', scores_std)
            # mlflow.log_metric('sharpe_mean', np.mean(sharpe))
            # mlflow.log_metric('sortino_mean', np.mean(sortino))
            # mlflow.log_metric('calmar_mean', np.mean(calmar))

            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: objective(trial, data, params['model_building']['index_base'], params['model_building']['indexes_higher'], params['model_building']['timeframes'], params['model_building']['timeframe_scalers'], params['model_building']['list_X'], params['model_building']['col_y'], cutoff_date, unique_dates, mondays_indexes, splits_all, experiment_id), n_trials=params['model_building']['n_trials'])
            print("Best trial number: ", study.best_trial.number)
            print("Best parameters: ", study.best_params)
            print("Best score:", study.best_value)
            mlflow.log_param('num_splits', len(splits_all[study.best_trial.number]))

            # run = mlflow.active_run()
            mlflow.log_metric('best_trial_number', study.best_trial.number)
            mlflow.log_dict(study.best_params, artifact_file="best_params.json")
            mlflow.log_metric('best_score', study.best_value)

            study_trials = study.trials_dataframe().sort_values(by=['value'], ascending=False)
            study_trials.to_csv("study_trials.csv", index=False)
            mlflow.log_artifact("study_trials.csv")
            mlflow.log_param('end_time', datetime.now())
            # mlflow.end_run(run_id=run.info.run_id)

        # Save the trained model in the root directory
        save_model(study.best_params, os.path.join(params['model_building']['models_path'], params['model_building']['model_params_name'].format(timeframe=params['model_building']['timeframe'], version=params['model_building']['version'])))


            # except Exception as e:
            #     logger.error(f"Failed to complete model building & evaluation: {e}")
            #     print(f"Error: {e}")


    except Exception as e:
        logger.error('Failed to complete the feature engineering and model building process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()

