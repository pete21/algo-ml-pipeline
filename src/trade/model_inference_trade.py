import logging
import json
import os
import urllib.error
import urllib.request
import dvc.api
import pandas as pd
import numpy as np
from src.data_utils.utils import getXy
from datetime import date, datetime
from dotenv import load_dotenv


load_dotenv()

PREDICT_URL = os.getenv("MODEL_PREDICT_URL", "http://localhost:8100/predict")
MODEL_PARAMS_URL = os.getenv("MODEL_PARAMS_URL", "http://localhost:8100/model/params")



def load_data(data_path: str, params: dict, logger: logging.Logger) -> dict:
    """Load data from a CSV file."""
    try:
        data = {}
        
        for i in params['model_building']['indexes_higher'] + [params['model_building']['index_base']]:
            filename = os.path.join(data_path, params['model_inference_trade']['file_name'].format(timeframe=params['model_building']['timeframes'][i]))
            print("Loading data from: ", filename)
            data[i] = pd.read_csv(filename, parse_dates=True, index_col='date')
        
        for i in params['model_building']['indexes_higher']:
            data[i]["date_merge"] = pd.to_datetime(data[i]["date_merge"])
        
        data[params['model_building']['index_base']]["date_merge"] = data[params['model_building']['index_base']].index

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
        logger.error('Prediction request failed with status %s: %s', exc.code, error_body)
        raise RuntimeError(
            f"Prediction request failed with status {exc.code}: {error_body}"
        ) from exc


def fetch_model_params(url: str, logger: logging.Logger) -> dict:
    """Fetch model_params from the model serving endpoint."""
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8")
        logger.error('Model params request failed with status %s: %s', exc.code, error_body)
        raise RuntimeError(
            f"Model params request failed with status {exc.code}: {error_body}"
        ) from exc


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
        print(f"Loaded model params: {model_params}")
        model_params['hour_range_start'] = 2
        model_params['hour_range_stop'] = 20

        p={}
        for i in params['model_building']['indexes_higher']:
            p[i] = model_params
        
        print(data[params['model_building']['index_base']].tail())
        print(data[params['model_building']['indexes_higher'][0]].tail())
        print(data[params['model_building']['indexes_higher'][1]].tail())

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

        y_pred_expected = np.matmul(predictions['predictions'], np.array([[-1],[0],[1]]))
        # y_series = pd.Series(y_pred-1, index=X_test.index, name="y_pred")
        # y_series = pd.Series(y_pred_expected.flatten(), index=X_test.index, name="y_pred").rolling(window=params['pred_avg_period'], min_periods=1).mean()
        y_series = pd.Series(y_pred_expected.flatten(), index=X.index[-num_rows:], name="y_pred").ewm(span=model_params['pred_ewm_span'], adjust=False).mean()
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
    result.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'result.csv'), index=True)

    print(result)
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
