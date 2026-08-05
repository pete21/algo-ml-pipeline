import logging
import os

# from src.data_utils.utils import load_params
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
        data[params['index_base']] = pd.read_csv(os.path.join(params['data_path'], params['file_name'].format(timeframe=params['timeframes'][params['index_base']])), parse_dates=True, index_col='date')
        data[params['index_base']]["high_time"] = pd.to_datetime(data[params['index_base']]["high_time"])
        data[params['index_base']]["low_time"] = pd.to_datetime(data[params['index_base']]["low_time"])
        for i in params['indexes_higher']:
            data[i] = pd.read_csv(os.path.join(params['data_path'], params['file_name'].format(timeframe=params['timeframes'][i])), parse_dates=True, index_col='date')
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

    # parameters = {'n_estimators': 382, 'max_depth': 7, 'learning_rate': 0.02, 'subsample': 0.95, 'gamma': 0.95, 'sma1_period': 5, 'sma2_period': 88, 'bb_periods': 18, 'bb_nbdev': 2.4625343800271473, 'ema1_period': 6, 'ema2_period': 19, 'sar_acc': 0.35347378805149954, 'sar_max': 1.102284705275549, 'midprice_window': 2, 'l1_fast': 8, 'l2_fast': 2, 'l3_fast': 8, 'l1_slow': 12, 'l2_slow': 3, 'l3_slow': 24, 'kama_trend_period': 25, 'ha_candle_period': 19, 'dc_market_regime_period': 19, 'displacement_strength_period': 35, 'displacement_strength': 1.0090179145664906, 'displacement_hull_period': 17, 'displacement_hull_slope_period': 14, 'gap_lookback': 6, 'gap_hull_period': 36, 'gap_hull_slope_period': 12, 'market_regime_threshold': 0.002987136210827556, 'tenkan_window': 7, 'kijun_window': 17, 'cci_timeperiods': 7, 'macd_fastperiod': 17, 'macd_slowperiod': 31, 'macd_signalperiod': 8, 'price_distribution_window_size': 5, 'price_distribution_percentile_threshold': 0.2, 'rsi_period': 27, 'rsi_slope_period': 6, 'stoch_fastk_period': 2, 'stoch_slowk_period': 4, 'stoch_slowd_period': 20, 'ppo_fastperiod': 3, 'ppo_slowperiod': 29, 'stochrsi_timeperiod': 7, 'stochrsi_fastk_period': 10, 'stochrsi_fastd_period': 13, 'train_range_len': 14, 'test_range_len': 5, 'hour_range_start': 10, 'hour_range_stop': 20, 'adx_timeperiod': 5, 'di_timeperiod': 19, 'macd_slope_period': 9, 'sl': 0.002294238807988796, 'tp': 0.002824953636382562, 'atr_period': 3, 'stochrsik_slope_period': 3, 'stochk_slope_period': 6, 'willr_timeperiod': 19, 'ha_sign_ma_period': 6, 'target_tp': 0.0024486199621331335, 'ema_period': 19, 'ema_reversed_period': 7, 'threshold_long': 0.8096038049250767, 'threshold_short': 0.17787209993585473}
    # p={}
    # p[7] = parameters
    # p[10] = parameters

    # data[params['index_base']] = calculate_features(data[params['index_base']], parameters, params['timeframe_scalers'][params['index_base']], col_close="Close", col_high="High", col_low="Low")
    # for i in params['indexes_higher']:
    #     data[i] = calculate_features(data[i], p[i], params['timeframe_scalers'][i], col_close="Close", col_high="High", col_low="Low")

    # return data


def save_data(data: dict, params: dict) -> None:
    """Save the processed dataset."""
    try:     
        os.makedirs(params['data_path_dest'], exist_ok=True)  # Ensure the directory is created
        logger.debug(f"Directory {params['data_path_dest']} created or already exists")

        for i in params['indexes_higher']:
            data[i].drop(columns=['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'minute_of_day', 'local_date'], inplace=True)
            data[i].to_csv(os.path.join(params['data_path_dest'], f'data_static_features_{params['timeframes'][i]}.csv'), index=True)

        data[params['index_base']].drop(columns=['local_date'], inplace=True)
        data[params['index_base']].to_csv(os.path.join(params['data_path_dest'], f"data_static_features_{params['timeframes'][params['index_base']]}.csv"), index=True)
        data[params['index_base']].isnull().sum().to_csv(os.path.join(params['data_path_dest'], 'nulls.csv'))
        data[params['index_base']].isin([np.inf, -np.inf]).sum().to_csv(os.path.join(params['data_path_dest'], 'inf.csv'))

        logger.debug(f"Processed data saved to {params['data_path_dest']}")
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
