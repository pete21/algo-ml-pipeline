import logging
import os
from datetime import date, datetime

import dvc.api
import numpy as np
import pandas as pd
import pytz
import requests
from dotenv import load_dotenv

from src.backtesting.optimization import drop_ohlc_columns
from src.data_utils.utils import getXy

load_dotenv()

PREDICT_URL = os.getenv("MODEL_PREDICT_URL", "http://localhost:8100/predict")
MODEL_PARAMS_URL = os.getenv("MODEL_PARAMS_URL", "http://localhost:8100/model/params")
MODEL_INFO_URL = os.getenv("MODEL_INFO_URL", "http://localhost:8100/")



def load_data(data_path: str, params: dict, logger: logging.Logger) -> dict:
    """Load data from a CSV file."""
    try:
        data = {}
        
        for i in params['model_building']['indexes_higher'] + [params['model_building']['index_base']]:
            filename = os.path.join(data_path, params['model_inference_trade']['file_name'].format(timeframe=params['model_building']['timeframes'][i]))
            print("Loading data from: ", filename)
            data[i] = pd.read_csv(filename, parse_dates=True, index_col='date')
        
        local_timezone = pytz.timezone(params['model_building']['local_timezone'])
        data[params['model_building']['index_base']]['local_date'] = data[params['model_building']['index_base']].index.tz_localize('UTC').tz_convert(local_timezone)


        data[params['model_building']['index_base']]["date_merge"] = data[params['model_building']['index_base']].index
        for i in params['model_building']['indexes_higher']:
            data[i]["date_merge"] = (
                data[i].index
                + pd.to_timedelta(params['model_building']['timeframe_minutes'][i], "m")
                - pd.to_timedelta(params['model_building']['timeframe_minutes'][params['model_building']['index_base']], "m")
            )

        logger.debug('Data loaded from %s', data_path)
        return data
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def request_predictions(X: pd.DataFrame, url: str, n_rows: int, logger: logging.Logger) -> dict:
    """POST the last n rows of X to the model serving /predict endpoint."""
    sample = X.tail(n_rows)
    payload = {
        "columns": sample.columns.tolist(),
        "data": sample.values.tolist(),
    }
    try:
        response = requests.post(
            url,
            json=payload
        )
        return response.json()
    except requests.exceptions.RequestException as exc:
        logger.error("Prediction request failed: %s", exc)
        raise


def fetch_model_params(url: str, logger: logging.Logger) -> dict:
    """Fetch model_params from the model serving endpoint."""
    response = requests.get(url)
    try:
        return response.json()
    except requests.exceptions.RequestException as exc:
        logger.error("Model params request failed: %s", exc)
        raise

def fetch_model_info(logger: logging.Logger) -> dict:
    """Fetch model info from the model serving endpoint."""
    response = requests.get(MODEL_INFO_URL)
    try:
        return response.json()
    except requests.exceptions.RequestException as exc:
        logger.error("Model info request failed: %s", exc)
        raise


def main(logger: logging.Logger) -> pd.DataFrame | None:
    print("Starting model inference process...")
    print(f"Start time: {datetime.now()}")
    y_series = None
    try:
        # Get root directory and resolve the path for params.yaml
        # root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')

        # Load parameters from the root directory
        params = dvc.api.params_show('params.yaml')
        # model_params = load_json_params(os.path.join(root_dir, 'model_params.json'), logger=logger)

        # Load the preprocessed data from the interim directory
        data = load_data(data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'), params=params, logger=logger)
        data[params['model_building']['index_base']].loc[:,"target"] = 0

        model_params = fetch_model_params(url=MODEL_PARAMS_URL, logger=logger)
        print(f"Model params: {model_params}")
        # model_params['hour_range_start'] = 6*60
        model_params['hour_range_stop'] = 18*60

        p={}
        for i in params['model_building']['indexes_higher']:
            p[i] = model_params
            # print(data[i].tail())

        
        X, _, columns = getXy(data,
        params['model_building']['index_base'],
        params['model_building']['indexes_higher'],
        model_params,
        p,
        params['model_building']['timeframes'],
        params['model_building']['timeframe_scalers'],
        params['model_building']['list_X'],
        'target',
        date.today()-pd.Timedelta(30, "d"),
        params['model_building']['lags'],
        col_open="Open", col_high="High", col_low="Low", col_close="Close"
        )

        print(X.columns)
        X = drop_ohlc_columns(X, params['model_building']['list_X'])
        print(X.columns)

        # X.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'X.csv'), index=True)
        print(X.tail())
        # print(y.head())
        # print(columns)
        num_rows = 10
        predictions = request_predictions(X, url=PREDICT_URL, n_rows=num_rows, logger=logger)
        print(f"Received {len(predictions.get('predictions', []))} predictions")
        print(predictions)
        # for i in predictions['predictions']:
            # print(i)
            # average = np.multiply(i, [-1,0,1])
            # print(np.sum(average))

        # y_pred_expected = np.matmul(predictions['predictions'], np.array([[-1],[0],[1]]))
        # y_series = pd.Series(y_pred-1, index=X_test.index, name="y_pred")
        # y_series = pd.Series(y_pred_expected.flatten(), index=X_test.index, name="y_pred").rolling(window=params['pred_avg_period'], min_periods=1).mean()
        y_series = pd.Series(predictions['predictions'], index=X.index[-num_rows:], name="y_pred").ewm(span=model_params['pred_ewm_span'], adjust=False).mean()
        print(y_series)

    except Exception as e:
        logger.error('Failed to complete the feature engineering and inference process: %s', e)
        print(f"Error: {e}")
        return None
    print("Inference process completed successfully.")
    print(f"End time: {datetime.now()}")
    
    result = X.tail(num_rows).join(y_series)
    # index_base = params['model_building']['index_base']
    # result['Close'] = data[index_base].reindex(result.index)['Close']
    result['tp'] = model_params['tp']
    result['sl'] = model_params['sl']
    result['raw_prediction'] = predictions['predictions']
    result.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'result.csv'), index=True)

    return result


if __name__ == '__main__':

    # logging configuration
    logger = logging.getLogger('trade')
    logger.setLevel('DEBUG')

    console_handler = logging.StreamHandler()
    console_handler.setLevel('DEBUG')

    file_handler = logging.FileHandler('trade_agent_log.log')
    file_handler.setLevel('ERROR')

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    main(logger)
