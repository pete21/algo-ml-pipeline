import logging
import os

import dvc.api
import numpy as np
import pandas as pd
import pytz

from src.data_utils.static_features import static_features

# logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('preprocessing_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(params: dict) -> dict:
    """Load the data from the data/raw directory."""
    try:
        data = {}
        data[params['index_base']] = pd.read_csv(os.path.join(params['data_path'], params['ticker'], params['file_name'].format(timeframe=params['timeframes'][params['index_base']])), parse_dates=True, index_col='date')
        data[params['index_base']]["high_time"] = pd.to_datetime(data[params['index_base']]["high_time"])
        data[params['index_base']]["low_time"] = pd.to_datetime(data[params['index_base']]["low_time"])
        for i in params['indexes_higher']:
            data[i] = pd.read_csv(os.path.join(params['data_path'], params['ticker'], params['file_name'].format(timeframe=params['timeframes'][i])), parse_dates=True, index_col='date')
            # data[i]["date_merge"] = pd.to_datetime(data[i]["date_merge"])
        return data
    except Exception as e:
        logger.error('Failed to load the data: %s', e)
        raise


def preprocess_data(data: dict, params: dict) -> dict:
    """Preprocess the data by adding static features"""
    try:
        local_timezone = pytz.timezone(params['local_timezone'])
        for i in [params['index_base']] + params['indexes_higher']:
            # data[i] = data[i].loc[data[i].index.date >= pd.to_datetime('2026-06-01').date()]
            data[i]['local_date'] = data[i].index.tz_localize('UTC').tz_convert(local_timezone)
            data[i] = static_features(data[i], params['timeframe_scalers'][i], high_col="High", low_col="Low", open_col="Open", close_col="Close")
            # print(data[i].head())

        return data
    except Exception as e:
        logger.error('Failed to preprocess the data: %s', e)
        raise


def save_data(data: dict, params: dict) -> None:
    """Save the processed dataset."""
    try:     
        os.makedirs(f"{params['data_path_dest']}/{params['ticker']}", exist_ok=True)  # Ensure the directory is created
        logger.debug(f"Directory {params['data_path_dest']}/{params['ticker']} created or already exists")

        for i in params['indexes_higher']:
            data[i].drop(columns=['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'minute_of_day', 'local_date'], inplace=True)
            data[i].to_csv(os.path.join(params['data_path_dest'], params['ticker'], 'data_static_features_{timeframe}.csv'.format(timeframe=params['timeframes'][i])), index=True)

        data[params['index_base']].drop(columns=['local_date'], inplace=True)
        data[params['index_base']].to_csv(os.path.join(params['data_path_dest'], params['ticker'], 'data_static_features_{timeframe}.csv'.format(timeframe=params['timeframes'][params['index_base']])), index=True)
        data[params['index_base']].isnull().sum().to_csv(os.path.join(params['data_path_dest'], params['ticker'], 'nulls.csv'))
        data[params['index_base']].isin([np.inf, -np.inf]).sum().to_csv(os.path.join(params['data_path_dest'], params['ticker'], 'inf.csv'))

        logger.debug(f"Processed data saved to {params['data_path_dest']}/{params['ticker']}")
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise

def main():
    try:
        logger.debug("Starting data preprocessing...")

        # params = load_params(params_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../params.yaml'), logger=logger)
        # Load parameters from the params.yaml in the root directory
        params = dvc.api.params_show('params.yaml')['data_preprocessing']
        print(f"Params: {params}")

        # Fetch the data from data/raw
        data = load_data(params=params)
        logger.debug('Data loaded successfully')

        # Preprocess the data
        data = preprocess_data(data, params)

        # Save the processed data
        save_data(data, params)
    except Exception as e:
        logger.error('Failed to complete the data preprocessing process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
