from yaml import YAMLError, safe_load
import logging
import numpy as np
import pandas as pd
import json
from json import JSONDecodeError
from datetime import date
from src.data_utils.dynamic_features import dynamic_features
from feature_engine.timeseries.forecasting import LagFeatures
from src.data_utils.features import calc_kernel_pca

def load_params(params_path: str, logger: logging.Logger) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_json_params(params_path: str, logger: logging.Logger) -> dict:
    """Load parameters from a JSON file."""
    try:
        with open(params_path, 'r') as file:
            params = json.load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except JSONDecodeError as e:
        logger.error('JSON error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def get_dates(data: dict, index: int) -> tuple[list, list, list]:
    unique_dates = np.unique(data[index].index.date)
    unique_weekdates = []
    for d in unique_dates:
        if d.weekday()<5:
            unique_weekdates.append(d)
    print("unique_dates: ", len(unique_dates), "unique_weekdates: ", len(unique_weekdates))

    mondays_indexes = [i for i, n in enumerate(unique_dates) if n.weekday() == 0]
    num_mondays = sum(1 for i in unique_dates if i.weekday() == 0)
    return unique_dates, unique_weekdates, mondays_indexes


def getXy(data: dict, index_b: int, indexes_h: list, parameters: dict, p: dict, timeframes: list, scalers: dict, X_cols: list, y_col: str, cutoff_date: date, lags: list, col_open="Open", col_high="High", col_low="Low", col_close="Close") -> tuple[pd.DataFrame, pd.DataFrame, list]:
    cutoff_date_2 = cutoff_date - pd.Timedelta(14, "D")
    ml_data = {}
    ml_data[index_b] = dynamic_features(data[index_b], parameters, scalers[index_b], col_close=col_close, col_high=col_high, col_low=col_low)
    ml_data[index_b] = ml_data[index_b][X_cols + [y_col] + [col_open, col_high, col_low, "date_merge", 'dow_sin', 'dow_cos', 'hour_sin', 'hour_cos']].loc[ml_data[index_b].index.date>=cutoff_date_2]

    target = ml_data[index_b].loc[(ml_data[index_b].index.hour>=parameters['hour_range_start']) & (ml_data[index_b].index.hour<=parameters['hour_range_stop']), [y_col]]

    lag_f = LagFeatures(variables = X_cols + [col_open, col_high, col_low], periods=lags, drop_na=True)

    for i in indexes_h:
        ml_data[i] = dynamic_features(data[i], p[i], scalers[i], col_close=col_close, col_high=col_high, col_low=col_low)
        ml_data[i] = ml_data[i][X_cols + [col_open, col_high, col_low, "date_merge"]].loc[ml_data[i].index.date>=cutoff_date_2]
        # print(ml_data[i].columns.values)
        ml_data[i] = lag_f.fit_transform(ml_data[i]).add_suffix(f"_{timeframes[i]}")
        # print(ml_data[i].columns.values)

    # ml_data[index_b].isnull().sum().to_csv('nulls.csv')

    ml_data[index_b] = lag_f.fit_transform(ml_data[index_b])
    ml_data[index_b] = ml_data[index_b].loc[(ml_data[index_b].index.hour>=parameters['hour_range_start']) & (ml_data[index_b].index.hour<=parameters['hour_range_stop'])]
    # print(ml_data[index_b].columns.values)

    for i in indexes_h:
#        ml_data[i] = ml_data[i].loc[(ml_data[i][f"date_merge_{timeframes[i]}"].dt.hour>=parameters['hour_range_start']) & (ml_data[i][f"date_merge_{timeframes[i]}"].dt.hour<=parameters['hour_range_stop'])]

#        ml_data[i] = ml_data[i].loc[(ml_data[i].index.hour>=parameters['hour_range_start']) & (ml_data[i].index.hour<=parameters['hour_range_stop'])]
#        ml_data[index_b] = ml_data[index_b].merge(ml_data[i], how='left', left_index=True, right_index=True)

        ml_data[index_b] = pd.merge_ordered(
            ml_data[index_b],
            ml_data[i],
            fill_method="ffill",
            left_on="date_merge",
            right_on=f"date_merge_{timeframes[i]}",
            how="left"
        )

    ml_data[index_b].set_index('date_merge', drop=True, inplace=True)
    ml_data[index_b] = ml_data[index_b].loc[ml_data[index_b].index.date>=cutoff_date]
    # ml_data[index_b].to_csv('ml_data_index_b.csv')
    # print(ml_data[index_b].columns.values)

#    ml_data[index_b][ml_data[index_b].select_dtypes(np.float16).columns] = ml_data[index_b].select_dtypes(np.float16).astype(np.float32)
    ml_data[index_b][ml_data[index_b].select_dtypes(np.float64).columns] = ml_data[index_b].select_dtypes(np.float64).astype(np.float32)
    ml_data[index_b][ml_data[index_b].select_dtypes(np.int64).columns] = ml_data[index_b].select_dtypes(np.int64).astype(np.int32)

    # feature_columns = lag_f.get_feature_names_out()
    # feature_columns.remove(y_col)
    # feature_columns.remove("date_merge")
    feature_columns = [x for x in lag_f.get_feature_names_out() if x not in ([y_col] + ["date_merge"])]           # [col_open, col_high, col_low, col_close] - exclude open, high, low, close

    X_columns = []
    for x in feature_columns:
        X_columns.append(x)
        for i in indexes_h:

#             m_search = re.search('(?:slope|log)_(\d+)', x, flags=re.ASCII)
#             if m_search:
#                 m=m_search.group(1)
#                 if timeframe_minutes[i]*int(m) > 240:
# #                    print(f'Skipped: {x},{timeframes[i]},lag {m}')
#                     continue
            if x not in ['dow_sin', 'dow_cos', 'hour_sin', 'hour_cos']:
                X_columns.append(f"{x}_{timeframes[i]}")

    dates = np.unique(ml_data[index_b].index.date)
# PCA
    # print('log_ret_ha_short_pca')                      # ['log_ret_ha_short_pca1','log_ret_ha_short_pca2']
    # cols = [x for x in X_columns if x.startswith("ret_ha_log") and (x[-1].isdigit() or x.endswith("15m"))]
    # pca_res = calc_kernel_pca(ml_data[index_b], dates[1:], 10, cols, ['log_ret_ha_short_pca1','log_ret_ha_short_pca2'])
    # ml_data[index_b]=ml_data[index_b].join(pca_res)
    # X_columns.append('log_ret_ha_short_pca1')
    # X_columns.append('log_ret_ha_short_pca2')
# X_columns = [x for x in X_columns if x not in cols]

    # print('log_ret_ha_long_pca')
    # cols = [x for x in X_columns if x.startswith("ret_ha_log") and (x.endswith("1h"))]
    # pca_res = calc_kernel_pca(ml_data[index_b], dates[1:], 10, cols, ['log_ret_ha_long_pca1','log_ret_ha_long_pca2'])
    # ml_data[index_b]=ml_data[index_b].join(pca_res)
    # X_columns.append('log_ret_ha_long_pca1')
    # X_columns.append('log_ret_ha_long_pca2')
# X_columns = [x for x in X_columns if x not in cols]

    if parameters['pca_ichimoku']:
        print('ichimoku_short_pca')
        cols = [x for x in X_columns if (x.startswith("tenkan_sen") or x.startswith("kijun_sen")) and not(x.endswith("1h"))]
        pca_res = calc_kernel_pca(ml_data[index_b], dates[1:], 10, cols, ['ichimoku_short_pca1','ichimoku_short_pca2'])
        ml_data[index_b]=ml_data[index_b].join(pca_res)
        X_columns.append('ichimoku_short_pca1')
        X_columns.append('ichimoku_short_pca2')
    #    X_columns = [x for x in X_columns if x not in cols]

        print('ichimoku_long_pca')
        cols = [x for x in X_columns if (x.startswith("tenkan_sen") or x.startswith("kijun_sen")) and (x.endswith("1h"))]
        pca_res = calc_kernel_pca(ml_data[index_b], dates[1:], 10, cols, ['ichimoku_long_pca1','ichimoku_long_pca2'])
        ml_data[index_b]=ml_data[index_b].join(pca_res)
        X_columns.append('ichimoku_long_pca1')
        X_columns.append('ichimoku_long_pca2')
    #    X_columns = [x for x in X_columns if x not in cols]
        cols=[
            'tenkan_sen','tenkan_sen_lag_1','tenkan_sen_lag_2',
            'tenkan_sen_15m','tenkan_sen_lag_1_15m','tenkan_sen_lag_2_15m',
            'tenkan_sen_1h','tenkan_sen_lag_1_1h','tenkan_sen_lag_2_1h',
            'kijun_sen','kijun_sen_lag_1','kijun_sen_lag_2',
            'kijun_sen_15m','kijun_sen_lag_1_15m','kijun_sen_lag_2_15m',
            'kijun_sen_1h','kijun_sen_lag_1_1h','kijun_sen_lag_2_1h',
        ]
        X_columns = [x for x in X_columns if x not in cols]

    if parameters['pca_kama']:
        print('kama_short_pca')
        cols = [x for x in X_columns if (x.startswith("kama_trend_slow_diff") or x.startswith("kama_trend_fast_diff")) and not (x.endswith("1h"))]
        pca_res = calc_kernel_pca(ml_data[index_b], dates[1:], 10, cols, ['kama_short_pca1','kama_short_pca2'])
        ml_data[index_b]=ml_data[index_b].join(pca_res)
        X_columns.append('kama_short_pca1')
        X_columns.append('kama_short_pca2')
    #    X_columns = [x for x in X_columns if x not in cols]

        print('kama_long_pca')
        cols = [x for x in X_columns if (x.startswith("kama_trend_slow_diff") or x.startswith("kama_trend_fast_diff")) and (x.endswith("1h"))]
        pca_res = calc_kernel_pca(ml_data[index_b], dates[1:], 10, cols, ['kama_long_pca1','kama_long_pca2'])
        ml_data[index_b]=ml_data[index_b].join(pca_res)
        X_columns.append('kama_long_pca1')
        X_columns.append('kama_long_pca2')
    #    X_columns = [x for x in X_columns if x not in cols]

        #print(X_columns)
        #print(len(X_columns))
        cols=[
            # 'sine','sine_lag_1','sine_lag_2',
            # 'sine_15m','sine_lag_1_15m','sine_lag_2_15m',
            # 'lowerband_r_1h','lowerband_r_lag_1_1h','lowerband_r_lag_2_1h',
            # 'upperband_r_1h','upperband_r_lag_1_1h','upperband_r_lag_2_1h',
            # 'ema_ha_wickstrength_15m','ema_ha_wickstrength_lag_1_15m','ema_ha_wickstrength_lag_2_15m',
            # 'stochrsid_1h','stochrsid_lag_1_1h','stochrsid_lag_2_1h',
            # 'rsi_1h','rsi_lag_1_1h','rsi_lag_2_1h',
            # 'rsi_ha_1h','rsi_ha_lag_1_1h','rsi_ha_lag_2_1h',
            # 'rsi','rsi_lag_1','rsi_lag_2',
            'kama_trend_slow_diff','kama_trend_slow_diff_15m','kama_trend_slow_diff_1h',
            'kama_trend_fast_diff','kama_trend_fast_diff_15m','kama_trend_fast_diff_1h',
            'kama_trend_slow_diff2','kama_trend_slow_diff2_15m','kama_trend_slow_diff2_1h',
            'kama_trend_fast_diff2','kama_trend_fast_diff2_15m','kama_trend_fast_diff2_1h',
            'kama_trend_slow_diff_lag_1','kama_trend_slow_diff_lag_1_15m','kama_trend_slow_diff_lag_1_1h',
            'kama_trend_fast_diff_lag_1','kama_trend_fast_diff_lag_1_15m','kama_trend_fast_diff_lag_1_1h',
            'kama_trend_slow_diff2_lag_1','kama_trend_slow_diff2_lag_1_15m','kama_trend_slow_diff2_lag_1_1h',
            'kama_trend_fast_diff2_lag_1','kama_trend_fast_diff2_lag_1_15m','kama_trend_fast_diff2_lag_1_1h',
            'kama_trend_slow_diff_lag_2','kama_trend_slow_diff_lag_2_15m','kama_trend_slow_diff_lag_2_1h',
            'kama_trend_fast_diff_lag_2','kama_trend_fast_diff_lag_2_15m','kama_trend_fast_diff_lag_2_1h',
            'kama_trend_slow_diff2_lag_2','kama_trend_slow_diff2_lag_2_15m','kama_trend_slow_diff2_lag_2_1h',
            'kama_trend_fast_diff2_lag_2','kama_trend_fast_diff2_lag_2_15m','kama_trend_fast_diff2_lag_2_1h',
        ]
        X_columns = [x for x in X_columns if x not in cols]

    # print(X_columns)
    X = ml_data[index_b][X_columns]
    y = ml_data[index_b][y_col]
    #print(lag_f.get_feature_names_out())
    X.fillna(0,inplace=True)
    return X, y, X_columns
