import mlflow
import logging
import optuna
import pandas as pd
import os
import json
from src.data_utils.utils import load_params, get_dates
from src.backtesting.optimization import objective
from datetime import date, datetime
from dotenv import load_dotenv

from src.model.mlflow_utils import create_mlflow_experiment, save_model_params

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

load_dotenv()


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
        root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')

        # Load parameters from the root directory
        params = load_params(os.path.join(root_dir, 'params.yaml'), logger=logger)
        # model_params = load_json_params(os.path.join(root_dir, 'model_params.json'), logger=logger)

        # Load the preprocessed data from the interim directory
        data = load_data(data_path=params['model_building']['data_path'], params=params)
        
        cutoff_date = date(2024,6,1)
        for d in data:
            data[d] = data[d].loc[data[d].index.date>=cutoff_date-pd.Timedelta(21, "D")]
        unique_dates, unique_weekdates = get_dates(data, params['model_building']['index_base'])

        experiment_name = f'xgb-dax-pipeline-v{params["model_building"]["version"]}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        if params['model_building']['evals_strategy']:
            experiment_name += '-evals'
        else:
            experiment_name += '-trading'
        print(f"Experiment name: {experiment_name}")

        experiment_id = create_mlflow_experiment(experiment_name=experiment_name, mlflow_tracking_uri=os.getenv('MLFLOW_TRACKING_URI'), tags={}, logger=logger)
        mlflow.set_experiment(experiment_id=experiment_id)
        # experiment = mlflow.get_experiment(experiment_id=experiment_id)

        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, data, params['model_building'], cutoff_date, unique_weekdates, experiment_id), n_trials=params['model_building']['n_trials'])
        print("Best trial number: ", study.best_trial.number)
        print("Best parameters: ", study.best_params)
        print("Best score:", study.best_value)


        # Save the trained model in the root directory
        print("Saving model parameters to json...")
        model_params_path = os.path.join(params['model_building']['models_path'], params['model_building']['model_params_name'].format(timeframe=params['model_building']['timeframe'], version=params['model_building']['version']))
        save_model_params(model_params=study.best_params, file_path=model_params_path, logger=logger)


        print("Saving study trials to csv...")
        optuna_trials_path = os.path.join(params['model_building']['models_path'],f"{experiment_name}_trials.csv")
        study_trials = study.trials_dataframe().sort_values(by=['value'], ascending=False)
        print("Study trials: ", study_trials.head())
        study_trials.to_csv(optuna_trials_path, index=False)
        study_trials.to_csv(os.path.join(params['model_building']['models_path'],"optuna_trials.csv"), index=False)

        print(f"Searching for run_id with run name: Trial_{study.best_trial.number}")
        # get run_id with run name 
        run_object = mlflow.search_runs(filter_string=f"run_name = 'Trial_{study.best_trial.number}'")
        run_id = run_object["run_id"][0]

        tags = {
            "project_name": "xgb-dax-pipeline",
            "stage": "building",
            "mlflow.note.content": params['model_building']['note'],
            "optimizer": "optuna",
            "model_family": "xgboost",
            "model_name": params['model_building']['model_name'],
            "best_trial_number": study.best_trial.number,
            "best_run_name": f"Trial_{study.best_trial.number}",
            "best_run_id": run_id,
        }
        print("Setting tags for experiment: ", tags)
        mlflow.set_experiment_tags(tags)

        with mlflow.start_run(run_id=run_id) as run:
            mlflow.log_artifact(local_path=os.path.join(params['model_building']['models_path'],"optuna_trials.csv"), artifact_path='optuna_trials')
            mlflow.log_artifact(local_path=model_params_path, artifact_path='model_params')
            mlflow.log_metric('best', True)
    
    except Exception as e:
        logger.error('Failed to complete the feature engineering and model building process: %s', e)
        print(f"Error: {e}")
    print("Model building process completed successfully.")


if __name__ == '__main__':
    main()
