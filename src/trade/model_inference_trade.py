import logging
import os
from datetime import date, datetime

import dvc.api
import pandas as pd
import pytz

# from src.backtesting.optimization import drop_ohlc_columns
from src.data_utils.utils import getXy
from src.trade.utils_model_serving import fetch_model_params, request_predictions


def load_data(data_path: str, params: dict, logger: logging.Logger) -> dict:
    """Load data from a CSV file."""
    try:
        data = {}
        
        for i in params['indexes_higher'] + [params['index_base']]:
            filename = os.path.join(data_path, params['file_name'].format(ticker=params['ticker'], timeframe=params['timeframes'][i]))
            print("Loading data from: ", filename)
            data[i] = pd.read_csv(filename, parse_dates=True, index_col='date')
        
        local_timezone = pytz.timezone(params['local_timezone'])
        data[params['index_base']]['local_date'] = data[params['index_base']].index.tz_localize('UTC').tz_convert(local_timezone)


        data[params['index_base']]["date_merge"] = data[params['index_base']].index
        for i in params['indexes_higher']:
            data[i]["date_merge"] = (
                data[i].index
                + pd.to_timedelta(params['timeframe_minutes'][i], "m")
                - pd.to_timedelta(params['timeframe_minutes'][params['index_base']], "m")
            )

        logger.debug('Data loaded from %s', data_path)
        return data
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise




def main(logger: logging.Logger) -> pd.DataFrame | None:
    print("Starting model inference process...")
    print(f"Start time: {datetime.now()}")
    y_series = None
    try:
        # Get root directory and resolve the path for params.yaml
        # root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')

        # Load parameters from the root directory
        params = dvc.api.params_show('params.yaml')['model_trade']
        # model_params = load_json_params(os.path.join(root_dir, 'model_params.json'), logger=logger)

        # Load the preprocessed data from the interim directory
        data = load_data(data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'), params=params, logger=logger)
        data[params['index_base']].loc[:,"target"] = 0

        model_params = fetch_model_params(logger=logger)
        print(f"Model params: {model_params}")

        p={}
        for i in params['indexes_higher']:
            p[i] = model_params
            # print(data[i].tail())

        
        X, _, columns = getXy(data,
        params['index_base'],
        params['indexes_higher'],
        model_params,
        p,
        params['timeframes'],
        params['timeframe_scalers'],
        params['list_X'],
        'target',
        date.today()-pd.Timedelta(30, "d"),
        params['lags'],
        col_open="Open", col_high="High", col_low="Low", col_close="Close"
        )

        model_params['hour_range_start'] = 0*60
        model_params['hour_range_stop'] = 20*60
        X = X.loc[(X['minute_of_day']>=model_params['hour_range_start']) & (X['minute_of_day']<model_params['hour_range_stop'])]                # limit trading hours

        # print(X.columns.values)
        to_drop = ['minute_of_day', 'local_date']
        X = X.drop(columns=to_drop)
        # X = drop_ohlc_columns(X, params['list_X'])                  # do not drop Close as it will be used to calculate order limit price
        # print(X.columns)

        # X.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'X.csv'), index=True)
        print(X.tail())
        # print(y.head())
        # print(columns)
        num_rows = 10
        predictions = request_predictions(X=X, n_rows=num_rows, logger=logger)
        print(f"Received {len(predictions.get('predictions', []))} predictions")
        print(predictions)
        # for i in predictions['predictions']:
            # print(i)
            # average = np.multiply(i, [-1,0,1])
            # print(np.sum(average))

        # y_pred_expected = np.matmul(predictions['predictions'], np.array([[-1],[0],[1]]))
        # y_series = pd.Series(y_pred-1, index=X_test.index, name="y_pred")
        # y_series = pd.Series(y_pred_expected.flatten(), index=X_test.index, name="y_pred").rolling(window=model_params['pred_avg_period'], min_periods=1).mean()
        y_series = pd.Series(predictions['predictions'], index=X.index[-num_rows:], name="y_pred").ewm(span=model_params['pred_ewm_span'], adjust=False).mean()
        print(y_series)

    except Exception as e:
        logger.error('Failed to complete the feature engineering and inference process: %s', e)
        print(f"Error: {e}")
        return None
    print("Inference process completed successfully.")
    print(f"End time: {datetime.now()}")
    
    result = X.tail(num_rows).join(y_series)
    # index_base = params['index_base']
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
