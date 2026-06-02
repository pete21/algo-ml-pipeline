import pandas as pd
import numpy as np
import os
import logging

from src.lib.static_features import static_features
from src.lib.utils import get_dates, load_params

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


def load_data(data_path: str, params: dict) -> dict:
    """Load the data from the data/raw directory."""
    try:
        data = {}
        data[params['data_preprocessing']['index_base']] = pd.read_csv(f'{data_path}/raw/data_ohlc_features_{params['data_preprocessing']['timeframes'][params['data_preprocessing']['index_base']]}.csv', parse_dates=True, index_col='date')
        data[params['data_preprocessing']['index_base']]["high_time"] = pd.to_datetime(data[params['data_preprocessing']['index_base']]["high_time"])
        data[params['data_preprocessing']['index_base']]["low_time"] = pd.to_datetime(data[params['data_preprocessing']['index_base']]["low_time"])
        for i in params['data_preprocessing']['indexes_higher']:
            data[i] = pd.read_csv(f'{data_path}/raw/data_ohlc_features_{params['data_preprocessing']['timeframes'][i]}.csv', parse_dates=True, index_col='date')
        for i in (params['data_preprocessing']['indexes_higher']+[params['data_preprocessing']['index_base']]):
            data[i]["date_merge"] = pd.to_datetime(data[i]["date_merge"])
        return data
    except Exception as e:
        logger.error('Failed to load the data: %s', e)
        raise



def preprocess_data(data: dict, params: dict) -> dict:
    """Preprocess the data by adding static features"""
    try:
        
        unique_dates, unique_weekdates, mondays_indexes = get_dates(data, params['data_preprocessing']['index_base'])
        
        for i in params['data_preprocessing']['indexes_higher']:
            data[i] = static_features(data[i], unique_weekdates, params['data_preprocessing']['timeframe_scalers'][i], high_col="High", low_col="Low", open_col="Open", close_col="Close")

        data[params['data_preprocessing']['index_base']] = static_features(data[params['data_preprocessing']['index_base']], unique_weekdates, params['data_preprocessing']['timeframe_scalers'][params['data_preprocessing']['index_base']], high_col="High", low_col="Low", open_col="Open", close_col="Close")


        return data
    except Exception as e:
        logger.error('Failed to preprocess the data: %s', e)
        raise

    # parameters = {'n_estimators': 382, 'max_depth': 7, 'learning_rate': 0.02, 'subsample': 0.95, 'gamma': 0.95, 'sma1_period': 5, 'sma2_period': 88, 'bb_periods': 18, 'bb_nbdev': 2.4625343800271473, 'ema1_period': 6, 'ema2_period': 19, 'sar_acc': 0.35347378805149954, 'sar_max': 1.102284705275549, 'midprice_window': 2, 'l1_fast': 8, 'l2_fast': 2, 'l3_fast': 8, 'l1_slow': 12, 'l2_slow': 3, 'l3_slow': 24, 'kama_trend_period': 25, 'ha_candle_period': 19, 'dc_market_regime_period': 19, 'displacement_strength_period': 35, 'displacement_strength': 1.0090179145664906, 'displacement_hull_period': 17, 'displacement_hull_slope_period': 14, 'gap_lookback': 6, 'gap_hull_period': 36, 'gap_hull_slope_period': 12, 'market_regime_threshold': 0.002987136210827556, 'tenkan_window': 7, 'kijun_window': 17, 'cci_timeperiods': 7, 'macd_fastperiod': 17, 'macd_slowperiod': 31, 'macd_signalperiod': 8, 'price_distribution_window_size': 5, 'price_distribution_percentile_threshold': 0.2, 'rsi_period': 27, 'rsi_slope_period': 6, 'stoch_fastk_period': 2, 'stoch_slowk_period': 4, 'stoch_slowd_period': 20, 'ppo_fastperiod': 3, 'ppo_slowperiod': 29, 'stochrsi_timeperiod': 7, 'stochrsi_fastk_period': 10, 'stochrsi_fastd_period': 13, 'train_range_len': 14, 'test_range_len': 5, 'hour_range_start': 10, 'hour_range_stop': 20, 'adx_timeperiod': 5, 'di_timeperiod': 19, 'macd_slope_period': 9, 'sl': 0.002294238807988796, 'tp': 0.002824953636382562, 'atr_period': 3, 'stochrsik_slope_period': 3, 'stochk_slope_period': 6, 'willr_timeperiod': 19, 'ha_sign_ma_period': 6, 'target_tp': 0.0024486199621331335, 'ema_period': 19, 'ema_reversed_period': 7, 'threshold_long': 0.8096038049250767, 'threshold_short': 0.17787209993585473}
    # p={}
    # p[7] = parameters
    # p[10] = parameters

    # data[params['data_preprocessing']['index_base']] = calculate_features(data[params['data_preprocessing']['index_base']], parameters, params['data_preprocessing']['timeframe_scalers'][params['data_preprocessing']['index_base']], col_close="Close", col_high="High", col_low="Low")
    # for i in params['data_preprocessing']['indexes_higher']:
    #     data[i] = calculate_features(data[i], p[i], params['data_preprocessing']['timeframe_scalers'][i], col_close="Close", col_high="High", col_low="Low")

    return data


def save_data(data: dict, params: dict, data_path: str) -> None:
    """Save the processed dataset."""
    try:
        interim_data_path = os.path.join(data_path, 'interim')
        logger.debug(f"Creating directory {interim_data_path}")
        
        os.makedirs(interim_data_path, exist_ok=True)  # Ensure the directory is created
        logger.debug(f"Directory {interim_data_path} created or already exists")

        for i in params['data_preprocessing']['indexes_higher']:
            data[i].to_csv(f'data_static_features_{params['data_preprocessing']['timeframes'][i]}.csv')

        data[params['data_preprocessing']['index_base']].to_csv(os.path.join(interim_data_path, f"data_static_features_{params['data_preprocessing']['index_base']}.csv"), index=False)
        data[params['data_preprocessing']['index_base']].isnull().sum().to_csv(os.path.join(interim_data_path, 'nulls.csv'))
        data[params['data_preprocessing']['index_base']].isin([np.inf, -np.inf]).sum().to_csv(os.path.join(interim_data_path, 'inf.csv'))

        logger.debug(f"Processed data saved to {interim_data_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise

def main():
    try:
        logger.debug("Starting data preprocessing...")
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data')

        params = load_params(params_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../params.yaml'), logger=logger)

        # Fetch the data from data/raw
        data = load_data(data_path=data_path+'/raw', params=params)
        logger.debug('Data loaded successfully')

        # Preprocess the data
        data = preprocess_data(data, params)

        # Save the processed data
        save_data(data, data_path=data_path)
    except Exception as e:
        logger.error('Failed to complete the data preprocessing process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
