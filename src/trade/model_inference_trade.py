
import logging
import json
import os
import urllib.error
import urllib.request
import pandas as pd
from src.data_utils.utils import getXy, load_params, get_dates
from src.backtesting.optimization import objective
from datetime import date, datetime
from dotenv import load_dotenv


load_dotenv()

PREDICT_URL = os.getenv("MODEL_PREDICT_URL", "http://localhost:8000/predict")


# logging configuration
logger = logging.getLogger('model_inference_trade')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_inference_trade_errors.log')
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


def request_predictions(X: pd.DataFrame, url: str = PREDICT_URL, n_rows: int = 10) -> dict:
    """POST the last n rows of X to the model serving /predict endpoint."""
    sample = X.tail(n_rows)
    payload = {
        "columns": sample.columns.tolist(),
        "data": sample.values.tolist(),
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8")
        raise RuntimeError(
            f"Prediction request failed with status {exc.code}: {error_body}"
        ) from exc


def main():
    print("Starting model inference trade process...")
    try:
        # Get root directory and resolve the path for params.yaml
        root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')

        # Load parameters from the root directory
        params = load_params(os.path.join(root_dir, 'params.yaml'), logger=logger)
        # model_params = load_json_params(os.path.join(root_dir, 'model_params.json'), logger=logger)

        # Load the preprocessed data from the interim directory
        data = load_data(data_path=params['model_building']['data_path'], params=params)
        
        X, y, columns = getXy(data, params['model_building']['index_base'], params['model_building']['indexes_higher'], params['model_building'], params['model_building']['timeframe_scalers'], params['model_building']['list_X'], params['model_building']['col_y'], date(2026,1,1), params['model_building']['lags'], col_open="Open", col_high="High", col_low="Low", col_close="Close")
        print(X.head())
        print(y.head())
        print(columns)

        predictions = request_predictions(X, n_rows=10)
        print(f"Received {len(predictions.get('predictions', []))} predictions")
        print(predictions)

    
    except Exception as e:
        logger.error('Failed to complete the feature engineering and model building process: %s', e)
        print(f"Error: {e}")
    print("Model building process completed successfully.")


if __name__ == '__main__':
    main()
