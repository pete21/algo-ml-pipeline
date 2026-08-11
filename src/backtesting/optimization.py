import json
import multiprocessing as mp
import os
from datetime import date, datetime
from random import random
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from joblib import dump

from src.backtesting.strategies import (
    do_backtest_Strategy_elasticnet_regression,
    do_backtest_Strategy_lasso_regression,
    do_backtest_Strategy_linear_regression,
    do_backtest_Strategy_logistic_regression,
    do_backtest_Strategy_ridge_regression,
    do_backtest_Strategy_svc_classification,
    do_backtest_Strategy_svr_regression,
    do_backtest_Strategy_xgb_classification,
    do_backtest_Strategy_xgb_regression,
)
from src.data_utils.features import (
    build_target,
    build_target_2,
    build_target_2b,
    build_target_2c,
    build_target_triple,
)

# from xgboost import DMatrix
from src.data_utils.utils import getXy
from src.model.mlflow_utils import save_model_params

MULTIPROCESSING_POOL = 12

# LOG_SPLITS_TABLE={}
# for num_splits in range(5,16):
#     LOG_SPLITS_TABLE[num_splits] = num_splits-1-(np.round(np.logspace(0,10,num=num_splits-1,base=0.93308)-0.5,3))*((num_splits-2)*2)

# Alternative logarithmic split table
# start=1000
# n=24
# step=50
# arr = [np.log(i) for i in range(start, start+n*step, step)]-np.log(1)
# target_min, target_max = 1, n
# scaled_arr = ((arr - arr.min()) * (target_max - target_min)) / (arr.max() - arr.min()) + target_min

LOG_SPLITS_TABLE = {
    2: [1.0],
    5: [1.0, 2.236, 3.22 , 4.   ],
    6: [1.0, 2.272, 3.344, 4.24 , 5.   ],
    7: [1.0, 2.29, 3.42, 4.4 , 5.25, 6.  ],
    8: [1.0, 2.308, 3.472, 4.516, 5.44 , 6.268, 7.   ],
    9: [1.0, 2.316, 3.52 , 4.598, 5.578, 6.46 , 7.272, 8.   ],
    10: [1.0, 2.328, 3.544, 4.664, 5.688, 6.616, 7.48 , 8.28 , 9.   ],
    11: [1.0, 2.332,  3.574,  4.708,  5.77 ,  6.742,  7.66 ,  8.506, 9.28 , 10.   ],
    12: [1.0, 2.34,  3.58,  4.76,  5.84,  6.86,  7.8 ,  8.68,  9.5 , 10.28, 11.  ],
    13: [1.0, 2.342,  3.596,  4.784,  5.906,  6.94 ,  7.93 ,  8.832, 9.712, 10.526, 11.274, 12.   ],
    14: [1.0, 2.344,  3.616,  4.816,  5.944,  7.024,  8.032,  8.968, 9.88 , 10.72 , 11.536, 12.28 , 13.   ],
    15: [1.0, 2.352,  3.626,  4.848,  5.992,  7.084,  8.124,  9.086, 10.022, 10.906, 11.738, 12.518, 13.272, 14.   ],
    16: [1.0, 2.344,  3.632,  4.864,  6.04 ,  7.132,  8.196,  9.204, 10.156, 11.052, 11.92 , 12.76 , 13.544, 14.272, 15.   ],
    17: [1.0, 2.35,  3.64,  4.87,  6.07,  7.18,  8.26,  9.28, 10.27, 11.2 , 12.1 , 12.94, 13.75, 14.53, 15.28, 16.  ],
    18: [1.0, 2.344,  3.656,  4.904,  6.088,  7.24 ,  8.328,  9.352, 10.376, 11.336, 12.232, 13.128, 13.96 , 14.76 , 15.56 , 16.296, 17.   ],
    19: [1.0, 2.36,  3.652,  4.91 ,  6.1  ,  7.256,  8.378,  9.432, 10.452, 11.438, 12.39 , 13.274, 14.158, 14.974, 15.79 , 16.538, 17.286, 18.   ],
    20: [1.0, 2.368,  3.664,  4.924,  6.148,  7.3  ,  8.416,  9.496, 10.54 , 11.548, 12.484, 13.42 , 14.32 , 15.184, 16.012, 16.804, 17.56 , 18.28 , 19.   ],
    21: [1.0, 2.368,  3.66 ,  4.952,  6.168,  7.346,  8.448,  9.55 , 10.614, 11.64 , 12.59 , 13.54 , 14.452, 15.326, 16.2  , 16.998, 17.796, 18.556, 19.278, 20.   ],
    22: [1.0, 2.36,   3.68 ,  4.96 ,  6.16 ,  7.36 ,  8.52 ,  9.6 ,  10.68,  11.72,  12.72,  13.68,  14.6 ,  15.52,  16.36,  17.2 ,  18.  ,  18.8 ,  19.56,  20.28, 21.  ],
    23: [1.0, 2.344,  3.688,  4.948,  6.208,  7.384,  8.56,   9.652, 10.744, 11.794, 12.802, 13.768, 14.734, 15.658, 16.54,  17.38,  18.22,  19.018, 19.816, 20.572, 21.286, 22.  ],
    24: [1.0, 2.364,  3.684,  4.96,   6.192,  7.424,  8.568,  9.712, 10.812, 11.868, 12.88,  13.892, 14.86,  15.784, 16.664, 17.544, 18.424, 19.216, 20.052, 20.8,   21.548, 22.296, 23.  ],
    25: [1.0, 2.38,   3.668,  4.956,  6.198,  7.44,   8.59,   9.74,  10.844, 11.902, 12.96,  13.972, 14.938, 15.904, 16.824, 17.698, 18.572, 19.446, 20.228, 21.056, 21.792, 22.574, 23.264, 24.  ],
    }

def flatten_concatenation(matrix):
    flat_list = []
    for row in matrix:
        flat_list += row
    return flat_list


def calc_aggregate_target(df, model_params: dict):
        # _, data[index_base].loc[:,"labeling_dual_ema1"], _ = build_target(data[index_base], \
        #     open_col='Open', high_col='High', low_col='Low', high_time_col="high_time", low_time_col="low_time", \
        #     tp=model_params['target_tp'], ema_period=model_params['ema_period'], ema_reversed_period=model_params['ema_reversed_period'], \
        #     threshold_long=model_params['threshold_long'], threshold_short=model_params['threshold_short'])
        df.loc[:,"labeling_dual_ema1"] = 0

        _, df.loc[:,"labeling_dual_ema2"], _ = build_target_triple(df, \
            open_col='Open', high_col='High', low_col='Low', high_time_col="high_time", low_time_col="low_time", \
            tp=model_params['target_tp'], ema_period=model_params['ema_period'], ema_reversed_period=model_params['ema_reversed_period'], \
            threshold_long=model_params['threshold_long'], threshold_short=model_params['threshold_short'])
        # data[index_base].loc[:,"labeling_dual_ema2"] = 0

# data[index_base].loc[:,"labeling_binary"], data[index_base].loc[:,"labeling_dual_ema"], data[index_base].loc[:,"labeling_multi"]

        # data[index_base] = build_target_2b(df=data[index_base], ref_column="Close_wavelet")        # ha_close # Close_wavelet
        df = build_target_2c(df=df, ref_column="Close_wavelet", periods=model_params['target3_periods'])        # ha_close # Close_wavelet
        df.loc[:,"labeling_multi"] = df.loc[:,"labeling_dual_ema1"]*model_params['target1_weight'] + df.loc[:,"labeling_dual_ema2"]*model_params['target2_weight'] + df.loc[:,"labeling_multi2"]*model_params['target3_weight']


def objective(trial, data: dict, params: dict, cutoff_date: date, unique_dates: list, experiment_id: str, model_params_override: dict | None = None) -> float:

    index_base = params['index_base']
    indexes_higher = params['indexes_higher']
    timeframes = params['timeframes']
    timeframe_scalers = params['timeframe_scalers']
    list_X = params['list_X']
    col_y = params['col_y']
    lags = params['lags']
    
    print("Trial: ", trial.number if trial else f"Evaluation_{params['run_name']}")
    run_name = f"Trial_{trial.number}" if trial else f"Evaluation_{params['run_name']}"
    # Set tracking URI
    print("Setting MLFlow tracking and experiment ID...")
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    mlflow.set_experiment(experiment_id=experiment_id)
    # print(f"Experiment ID: {experiment_id}")

    # mlflow.xgboost.autolog()

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param('cutoff_date', cutoff_date)
        mlflow.log_param('evals_strategy', params['evals_strategy'])
        mlflow.log_params(params)

        model_params = model_params_override or {

            'sma1_period': trial.suggest_int('sma1_period', 12, 12),
            'sma2_period': trial.suggest_int('sma2_period', 85, 86), 
            'bb_periods': trial.suggest_int('bb_periods', 43, 43),
            'bb_nbdev': trial.suggest_float('bb_nbdev', 1.95, 2.05),
            'ema1_period': trial.suggest_int('ema1_period', 4, 4),
            'ema2_period': trial.suggest_int('ema2_period', 22, 22),
            'sar_acc': trial.suggest_float('sar_acc', 0.46, 0.5), 
            'sar_max': trial.suggest_float('sar_max', 0.62, 0.68), 
            'midprice_window': trial.suggest_int('midprice_window', 2, 2), # 2,30
            'l1_fast': trial.suggest_int('l1_fast', 8, 9), # 15,3,10
            'l2_fast': trial.suggest_int('l2_fast', 6, 6), 
            'l3_fast': trial.suggest_int('l3_fast', 16, 16), 
            'l1_slow': trial.suggest_int('l1_slow', 37, 39), 
            'l2_slow': trial.suggest_int('l2_slow', 6, 6),
            'l3_slow': trial.suggest_int('l3_slow', 30, 30),
            'kama_trend_period': trial.suggest_int('kama_trend_period', 34, 34),

            'ha_candle_period': trial.suggest_int('ha_candle_period', 28, 28), 
            'dc_market_regime_period': trial.suggest_int('dc_market_regime_period', 34, 34), 
            'displacement_strength_period': trial.suggest_int('displacement_strength_period', 25, 26), 
            'displacement_strength': trial.suggest_float('displacement_strength', 1.45, 1.55),
            'displacement_hull_period': trial.suggest_int('displacement_hull_period', 20, 20), 
            #    'displacement_sma_period': trial.suggest_int('displacement_sma_period', 2, 30), 
            'displacement_hull_slope_period': trial.suggest_int('displacement_hull_slope_period', 5, 6),

            'gap_lookback': trial.suggest_int('gap_lookback', 2, 2),
            'gap_hull_period': trial.suggest_int('gap_hull_period', 14, 15),             # minimum 4
            'gap_hull_slope_period': trial.suggest_int('gap_hull_slope_period', 6, 7),

            'market_regime_threshold': trial.suggest_float('market_regime_threshold', 0.003, 0.0032),
            'tenkan_window': trial.suggest_int('tenkan_window', 4, 8), 
            'kijun_window': trial.suggest_int('kijun_window', 55, 55), 
            'cci_timeperiods': trial.suggest_int('cci_timeperiods', 25, 25),
            'macd_fastperiod': trial.suggest_int('macd_fastperiod', 12, 12), 
            'macd_slowperiod': trial.suggest_int('macd_slowperiod', 35, 35), 
            'macd_signalperiod': trial.suggest_int('macd_signalperiod', 10, 10),
            'price_distribution_window_size': trial.suggest_int('price_distribution_window_size', 5, 5),   # 5,50
            'price_distribution_percentile_threshold': trial.suggest_float('price_distribution_percentile_threshold', 0.2, 0.2), # 0.2,0.5
            'rsi_period': trial.suggest_int('rsi_period', 19, 19),
            'rsi_slope_period': trial.suggest_int('rsi_slope_period', 15, 16),
            'stoch_fastk_period': trial.suggest_int('stoch_fastk_period', 8, 8),
            'stoch_slowk_period': trial.suggest_int('stoch_slowk_period', 14, 15),
            'stoch_slowd_period': trial.suggest_int('stoch_slowd_period', 25, 25),
            'ppo_fastperiod': trial.suggest_int('ppo_fastperiod', 10, 16),
            'ppo_slowperiod': trial.suggest_int('ppo_slowperiod', 39, 41),

            'stochrsi_timeperiod': trial.suggest_int('stochrsi_timeperiod', 12, 12),
            'stochrsi_fastk_period': trial.suggest_int('stochrsi_fastk_period', 3, 3),
            'stochrsi_fastd_period': trial.suggest_int('stochrsi_fastd_period', 17, 18),
            'train_range_len': trial.suggest_int('train_range_len', 22, 23),
            'test_range_len': trial.suggest_int('test_range_len', 4, 4),  #3,5
            'hour_range_start': trial.suggest_int('hour_range_start', 480, 480, step=15),
            # 'hour_range_stop': trial.suggest_int('hour_range_stop', 20, 20),
            'adx_timeperiod': trial.suggest_int('adx_timeperiod', 5, 5),      #5,15
            'di_timeperiod': trial.suggest_int('di_timeperiod', 11, 11),
            'macd_slope_period': trial.suggest_int('macd_slope_period', 9, 9),
            # 'sl': trial.suggest_float('sl', 0.003, 0.004) if not params['evals_strategy'] else 0,
            'tp': trial.suggest_float('tp', 0.0031, 0.0034) if not params['evals_strategy'] else trial.suggest_int('tp', 50, 150),

            'atr_period': trial.suggest_int('atr_period', 7, 7),

            'stochrsik_slope_period': trial.suggest_int('stochrsik_slope_period', 14, 15),
            'stochk_slope_period': trial.suggest_int('stochk_slope_period', 8, 12),
            'willr_timeperiod': trial.suggest_int('willr_timeperiod', 27, 30),

            'ha_sign_ma_period': trial.suggest_int('ha_sign_ma_period', 11, 11),

            'target_tp': trial.suggest_float('target_tp', 0.0028, 0.0031),
            'ema_period': trial.suggest_int('ema_period', 21, 21),
            'ema_reversed_period': trial.suggest_int('ema_reversed_period', 2, 4),
            'threshold_long': trial.suggest_float('threshold_long', 0.8, 0.8, step=0.01),
            'threshold_short': trial.suggest_float('threshold_short', 0.2, 0.2, step=0.01),
            'threshold': trial.suggest_float('threshold', 0.45, 0.45, step=0.01),
            'pred_ewm_span': trial.suggest_float('pred_ewm_span', 1.7, 2.5, step=0.05),
            'pca_ichimoku': trial.suggest_categorical('pca_ichimoku', [False]),
            'pca_kama': trial.suggest_categorical('pca_kama', [False]),
            'weekday': trial.suggest_categorical('weekday', [0]),                     # 0: Monday, 2: Wednesday, 4: Friday
            'target1_weight': trial.suggest_float('target1_weight', 0, 0, step=0.1),      # 1-1.8
            'target2_weight': trial.suggest_float('target2_weight', 1.5, 2, step=0.05),
            'target3_weight': trial.suggest_float('target3_weight', 1.8, 2.5, step=0.05),
            "target3_periods": trial.suggest_int('target3_periods', 22, 22),


            # XGBoost parameters
            'n_estimators': trial.suggest_int('n_estimators', 420, 420, step=5),
            'max_depth': trial.suggest_int('max_depth', 8, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.015, step=0.001),
            'subsample': trial.suggest_float('subsample', 0.95, 0.95),
            'gamma':  trial.suggest_float('gamma', 0.9, 0.9),
            # 'feature_fraction':  trial.suggest_float('feature_fraction', 0.9, 1),
            # 'num_leaves':  trial.suggest_int('num_leaves', 10, 200),

            # SVR and SVC parameters
            'C': trial.suggest_float('C', 0.3, 0.4, step=0.05),
            'epsilon': trial.suggest_float('epsilon', 0.375, 0.375, step=0.025),    # 0.3,0.5
        }

        if not model_params_override:
            model_params['hour_range_stop'] = trial.suggest_int('hour_range_stop', model_params['hour_range_start'] + 5*60, model_params['hour_range_start'] + 5*60)

            if params['evals_strategy']:
                model_params['sl'] = trial.suggest_int('sl', model_params['tp'] // 1.5, model_params['tp'] // 1.5)
            else:
                model_params['sl'] = trial.suggest_float('sl', model_params['tp'], model_params['tp'])


        calc_aggregate_target(data[index_base], model_params)

        p={}
        for i in indexes_higher:
            p[i] = model_params
        X, y, X_columns = getXy(data, index_base, indexes_higher, model_params, p, timeframes, timeframe_scalers, list_X, col_y[0], cutoff_date, lags, col_open="Open", col_high="High", col_low="Low", col_close="Close")
        y=y+1

        # print(X_columns)
        # X.to_csv('X.csv', index=True, header=True)
        # y.to_csv('y.csv', index=True, header=True)
        num_splits = params['num_splits']

        # Split the dataset to test and train sets
        split_weekdays = [i for i, n in enumerate(unique_dates) if n.weekday() == model_params['weekday']]
        num_splits_weekdays = len(split_weekdays)

        splits_array = LOG_SPLITS_TABLE[num_splits]
        splits = [int(5 + (num_splits_weekdays-5) * (random()/2 + i) / num_splits) for i in splits_array] # range(1, num_splits)]
        train_splits = [split_weekdays[i] for i in splits]
        print(train_splits)

        print(datetime.now().strftime('%H:%M:%S'))

        mlflow.log_params(model_params)


        print(f"Saving model parameters of {run_name} in artifacts...")
        model_params_path = os.path.join(params['models_path'], 'model_params.json')
        with open(model_params_path, 'w') as file:
            json.dump(model_params, file)
        mlflow.log_artifact(local_path=model_params_path, artifact_path='model_params')

        mlflow.log_param('num_splits', num_splits)
        mlflow.log_param('train_splits', train_splits)
        # mlflow.log_param('X_columns', X_columns)                      # too long to log as param (6k bytes max)
        mlflow.log_param('y_col', col_y[0])
        mlflow.log_param('index_base', index_base)
        mlflow.log_param('indexes_higher', indexes_higher)

        tuples = []
        for i in train_splits:

            train_split = unique_dates[i]
            train_start_idx = unique_dates[max(i-model_params['train_range_len']*5, 1)]
            test_end_idx = unique_dates[min(i+model_params['test_range_len']*5, len(unique_dates)-5)]
            print(train_start_idx, train_split, test_end_idx)

            mask = (X['local_date'].dt.date>train_start_idx) & (X['local_date'].dt.date<=train_split) & (X['minute_of_day']>=model_params['hour_range_start']) & (X['minute_of_day']<model_params['hour_range_stop'])
            X_train = X.loc[mask]
            y_train = y.loc[mask]

            X_train, y_train = remove_outliers(X_train, y_train, params, threshold=1.005)
            X_train = drop_ohlc_columns(X_train, list_X)

            X_test = X.loc[(X['local_date'].dt.date>train_split) & (X['local_date'].dt.date<=test_end_idx)]
            # y_test = y.loc[(X['local_date'].dt.date>train_split) & (X['local_date'].dt.date<=test_end_idx)]
            y_test = None

            data_target = X_test.loc[:,['Open','High','Low','Close','minute_of_day']]
            data_target['sl']=model_params['sl']
            data_target['tp']=model_params['tp']
            
            # data_target['DaytradingExit'] = ((data_target.index.date != data_target.index.to_series().shift(periods=-1).dt.date) | (data_target.index.date != data_target.index.to_series().shift(periods=-2).dt.date))
            data_target['DaytradingExit'] = (data_target['minute_of_day'] >= 21*60-15) & (data_target['minute_of_day'] <= 21*60)

            X_test = X_test.loc[(X_test['minute_of_day']>=model_params['hour_range_start']) & (X_test['minute_of_day']<model_params['hour_range_stop'])]

            X_test = drop_ohlc_columns(X_test, list_X)

            # X_train.to_csv('X_train_'+str(i)+'.csv')
            # y_train.to_csv('y_train_'+str(i)+'.csv')
            #data_target = data_target.join(ml_data['atr']).ffill().bfill()
            tuples.append((params, X_train, y_train, X_test, y_test, data_target, model_params))         # exponential_growth(1, 0.02, num_splits-idx-1)

        results = []
        # with mp.Pool(10) as p:
        with mp.Pool(MULTIPROCESSING_POOL, maxtasksperchild=1) as p:
            results = p.starmap(run_backtest_strategy, tuples)
            p.close()
            p.join()
            # p.terminate()


        # results.sort(key=lambda res: res[1]['Start'])

        stats = [res[1] for res in results]
        optimisation_score=0.0

        #        y_pred_all = y_pred_all.combine_first(res[0])
        #    y_pred_all.to_csv('y_pred_all_opt1.csv')
        total_trades = sum(s['# Trades'] for s in stats)

        profits = [s['Equity Final [$]']-100000 for s in stats]
        total_profit = np.nansum(profits)

        print(f'Splits: {train_splits}')
        print(f'Profits: {profits} Sum: {total_profit}')

        if total_trades > 0:

            mlflow.log_metric('total_trades', total_trades)
            mlflow.log_metric('total_profit', total_profit)
            sharpe = [s['Sharpe Ratio'] for s in stats]
            sortino = [s['Sortino Ratio'] for s in stats]
            # calmar = [s['Calmar Ratio'] for s in stats]
            sharpe_mean = np.nanmean(sharpe)
            sortino_mean = np.nanmean(sortino)
            # calmar_mean = np.nanmean(calmar)

            print(f'Sharpe: {sharpe} Avg: {sharpe_mean}')
            print(f'Sortino: {sortino} Avg: {sortino_mean}')
            # print(f'Calmar: {calmar} Avg: {calmar_mean}')

            mlflow.log_metric('sharpe_ratio', sharpe_mean)
            mlflow.log_metric('sortino_mean', sortino_mean)
            # mlflow.log_metric('calmar_mean', calmar_mean)

            total_return_percentage = total_profit/1000
            mlflow.log_metric('total_return_percentage', total_return_percentage)
            mlflow.log_metric('total_expectancy_percentage', total_return_percentage / total_trades)

            wins = [ s['_trades'].loc[s['_trades']['PnL']>0,'PnL'] for s in stats ]
            losses = [ s['_trades'].loc[s['_trades']['PnL']<0,'PnL'] for s in stats ]
            # draws = [ s['_trades'].loc[s['_trades']['PnL']==0,'PnL'] for s in stats ]
            
            wins_value = sum([w.sum(skipna=True) for w in wins])
            losses_value = sum([l.sum(skipna=True) for l in losses])
            win_trades = sum([w.count() for w in wins])
            loss_trades = sum([l.count() for l in losses])
            mlflow.log_metric('win_trades', win_trades)
            mlflow.log_metric('loss_trades', loss_trades)
            # mlflow.log_metric('draw_trades', sum([d.count() for d in draws]))
            
            avg_win = wins_value / win_trades if win_trades > 0 else 0
            avg_loss = losses_value / loss_trades if loss_trades > 0 else 0
            mlflow.log_metric('avg_win', avg_win)
            mlflow.log_metric('avg_loss', avg_loss)

            # win_trades = int(sum(np.where(np.isnan(s['Win Rate [%]']), 0, s['# Trades']*s['Win Rate [%]']/100) for s in stats))
            # loss_trades = total_trades - win_trades
            win_rate = win_trades / total_trades
            mlflow.log_metric('win_rate', win_rate)
            
            mlflow.log_metric('profit_factor', -wins_value / losses_value)

            sharpe_coeff = np.log(sharpe_mean) if sharpe_mean > 1 else sharpe_mean-1

            if params['evals_strategy']:
                optimisation_score = (total_profit + np.nanmean(np.sort(profits)[:int(num_splits/4)])) * (0.5 + sharpe_coeff/10 + win_rate + np.log(total_trades)/10 - avg_win/avg_loss/10)
            else:
                optimisation_score = (total_profit + np.nanmean(np.sort(profits)[:int(num_splits/4)])) * (1 + sharpe_coeff/10 + win_rate/5 + np.log(total_trades)/5 - avg_win/avg_loss/10) #* np.sqrt(max(np.mean(sortino)+np.mean(calmar), 1)) / (parameters['hour_range_stop']-parameters['hour_range_start']+2)
            
            # try:
            #     mlflow.log_metric('Avg_Drawdown_Duration_Seconds', int(sum(s['Avg. Drawdown Duration'].seconds*s['# Trades'] for s in stats) / total_trades))
            # except Exception as e:
            #     print(f"Error logging Avg_Drawdown_Duration_Seconds: {e}")
            #     mlflow.log_metric('Avg_Drawdown_Duration_Seconds', 0)

            for idx, res in enumerate(stats):
                with mlflow.start_run(nested=True, run_name=f"{run_name}_Child_Run_{idx}") as child_run:

                    try:
                        # mlflow.log_metric('Start', int(res[5]['Start'].timestamp()))
                        mlflow.log_metric('Start', res['Start'].year*10000+res['Start'].month*100+res['Start'].day)
                        # mlflow.log_metric('End', int(res[5]['End'].timestamp()))
                        mlflow.log_metric('End', res['End'].year*10000+res['End'].month*100+res['End'].day)

                        if res['# Trades']>0:

                            mlflow.log_metric('Equity_Final', res['Equity Final [$]'])
                            mlflow.log_metric('Equity_Peak', res['Equity Peak [$]'])
                            mlflow.log_metric('Commissions', res['Commissions [$]'])
                            mlflow.log_metric('Return_Percentage', res['Return [%]'])
                            mlflow.log_metric('Buy_Hold_Return_Percentage', res['Buy & Hold Return [%]'])
                            mlflow.log_metric('sharpe_ratio', res['Sharpe Ratio'])
                            mlflow.log_metric('Sortino_Ratio', res['Sortino Ratio'])
                            # mlflow.log_metric('Calmar_Ratio', res['Calmar Ratio'])
                            mlflow.log_metric('Max_Drawdown_Percentage', res['Max. Drawdown [%]'])
                            mlflow.log_metric('Avg_Drawdown_Percentage', res['Avg. Drawdown [%]'])
                            mlflow.log_metric('Max_Drawdown_Duration_Seconds', res['Max. Drawdown Duration'].seconds)
                            mlflow.log_metric('Avg_Drawdown_Duration_Seconds', res['Avg. Drawdown Duration'].seconds)
                            mlflow.log_metric('total_trades', res['# Trades'])
                            mlflow.log_metric('win_rate', res['Win Rate [%]']/100)
                            # mlflow.log_metric('Best_Trade_Percentage', res['Best Trade [%]'])
                            # mlflow.log_metric('Worst_Trade_Percentage', res['Worst Trade [%]'])
                            mlflow.log_metric('Avg_Trade_Percentage', res['Avg. Trade [%]'])
                            mlflow.log_metric('Max_Trade_Duration_Seconds', res['Max. Trade Duration'].seconds)
                            mlflow.log_metric('Avg_Trade_Duration_Seconds', res['Avg. Trade Duration'].seconds)
                            mlflow.log_metric('profit_factor', res['Profit Factor'])
                            mlflow.log_metric('Expectancy_Percentage', res['Expectancy [%]'])
                            mlflow.log_metric('SQN', res['SQN'])
                            mlflow.log_metric('Kelly_Criterion', res['Kelly Criterion'])

                            wins = res['_trades'].loc[res['_trades']['PnL']>0,'PnL']
                            losses = res['_trades'].loc[res['_trades']['PnL']<0,'PnL']
                            avg_win = wins.sum(skipna=True) / wins.count() if wins.count() > 0 else 0
                            avg_loss = losses.sum(skipna=True) / losses.count() if losses.count() > 0 else 0
                            mlflow.log_metric('avg_win', avg_win)
                            mlflow.log_metric('avg_loss', avg_loss)
                            mlflow.log_metric('win_trades', wins.count())
                            mlflow.log_metric('loss_trades', losses.count())

                            # draws = res['_trades'].loc[res['_trades']['PnL']==0, 'PnL']
                            # mlflow.log_metric('draw_trades', draws.count())

                            # TODO: Save res['_trades'] as artifact
                            # print(res['_trades'])
                            trades_path = os.path.join(params['models_path'], 'trades.csv')
                            res['_trades'].to_csv(trades_path, index=False)
                            mlflow.log_artifact(local_path=trades_path, artifact_path='trades')

                        else:
                            mlflow.log_metric('total_trades', res['# Trades'])
                            print(f"Child run {idx} has no trades")

                        # if idx == len(stats)-1:


                    except Exception as e:
                        # mlflow.log_params(res[5])
                        print(res)
                        print(f"Error logging child run {idx}: {e}")
            
        else:
            mlflow.log_metric('total_trades', 0)
            mlflow.log_metric('total_profit', 0)
            mlflow.log_metric('win_trades', 0)
            mlflow.log_metric('loss_trades', 0)
        
        mlflow.log_metric('optimisation_score', optimisation_score)

    return optimisation_score


def remove_outliers(X_df: pd.DataFrame, y_df: pd.DataFrame, params: dict, threshold: float = 1.005) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Remove outliers from X_train
    outliers = X_df.loc[X_df['High']/X_df['Low']>threshold].index
    extended_outliners = flatten_concatenation( [[outlier-pd.Timedelta(minutes=params['timeframe_minutes'][params['index_base']]), outlier, outlier+pd.Timedelta(minutes=params['timeframe_minutes'][params['index_base']]), outlier+pd.Timedelta(minutes=2*params['timeframe_minutes'][params['index_base']])] for outlier in outliers] )
    print('# outliers:', len(outliers))
    mask = ~X_df.index.isin(extended_outliners)
    X_df = X_df.loc[mask]
    y_df = y_df.loc[mask]

    return X_df, y_df

def drop_ohlc_columns(X_df: pd.DataFrame, list_X: list) -> pd.DataFrame:
    to_drop = ['minute_of_day', 'local_date']                # ['minute_of_day', 'local_date'] - dropped, not feature columns
    for col in ['Open', 'High', 'Low', 'Close']:             # ['Open', 'High', 'Low', 'Close'] - dropped with lags if not in list_X (features)
        if col not in list_X:                               
            to_drop.extend([s for s in X_df.columns if s.startswith(col)])
    return X_df.drop(columns=to_drop)

def run_backtest_strategy(params: dict, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, data_target: pd.DataFrame, model_params: dict) -> tuple[Any, Any]:
    if params['model_type'] == 'xgb_classification':
        model, scaler = do_backtest_Strategy_xgb_classification(X_train, y_train, X_test, y_test, data_target, model_params, params['evals_strategy'])
    elif params['model_type'] == 'xgb_regression':
        model, scaler = do_backtest_Strategy_xgb_regression(X_train, y_train, X_test, y_test, data_target, model_params, params['evals_strategy'])
    elif params['model_type'] == 'linear_regression':
        model, scaler = do_backtest_Strategy_linear_regression(X_train, y_train, X_test, y_test, data_target, model_params, params['evals_strategy'])
    elif params['model_type'] == 'logistic_regression':
        model, scaler = do_backtest_Strategy_logistic_regression(X_train, y_train, X_test, y_test, data_target, model_params, params['evals_strategy'])
    elif params['model_type'] == 'ridge_regression':
        model, scaler = do_backtest_Strategy_ridge_regression(X_train, y_train, X_test, y_test, data_target, model_params, params['evals_strategy'])
    elif params['model_type'] == 'lasso_regression':
        model, scaler = do_backtest_Strategy_lasso_regression(X_train, y_train, X_test, y_test, data_target, model_params, params['evals_strategy'])
    elif params['model_type'] == 'elasticnet_regression':
        model, scaler = do_backtest_Strategy_elasticnet_regression(X_train, y_train, X_test, y_test, data_target, model_params, params['evals_strategy'])
    elif params['model_type'] == 'svr_regression':
        model, scaler = do_backtest_Strategy_svr_regression(X_train, y_train, X_test, y_test, data_target, model_params, params['evals_strategy'])
    elif params['model_type'] == 'svc_classification':
        model, scaler = do_backtest_Strategy_svc_classification(X_train, y_train, X_test, y_test, data_target, model_params, params['evals_strategy'])
    else:
        raise ValueError(f"Invalid model type: {params['model_type']}")

    return model, scaler



def train_register_model(data: dict, params: dict, unique_weekdates: list, train_split_index: int, experiment_id: str, model_params: dict) -> str:

    index_base = params['index_base']
    indexes_higher = params['indexes_higher']
    timeframes = params['timeframes']
    timeframe_scalers = params['timeframe_scalers']
    list_X = params['list_X']
    col_y = params['col_y']
    lags = params['lags']
    
    # Set tracking URI
    print("Setting MLFlow tracking and experiment ID...")
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    mlflow.set_experiment(experiment_id=experiment_id)
    # print(f"Experiment ID: {experiment_id}")

    # mlflow.xgboost.autolog()
    run_name = f"Train_Register_Model_{unique_weekdates[train_split_index]}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)

        calc_aggregate_target(data[index_base], model_params)

        p={}
        for i in indexes_higher:
            p[i] = model_params
        X, y, X_columns = getXy(data, index_base, indexes_higher, model_params, p, timeframes, timeframe_scalers, list_X, col_y[0], date(2026,1,1), lags, col_open="Open", col_high="High", col_low="Low", col_close="Close")
        y=y+1

        X.to_csv('X_reg.csv', index=True, header=True)
        y.to_csv('y_reg.csv', index=True, header=True)

        train_splits = [train_split_index]
        print(train_splits)

        print(datetime.now().strftime('%H:%M:%S'))

        mlflow.log_params(model_params)
        mlflow.log_param('train_splits', train_splits)
        # mlflow.log_param('X_columns', X_columns)
        mlflow.log_param('y_col', col_y[0])
        mlflow.log_param('index_base', index_base)
        mlflow.log_param('indexes_higher', indexes_higher)


        train_split = unique_weekdates[train_splits[-1]]
        train_start_idx = unique_weekdates[max(train_splits[-1]-model_params['train_range_len']*5, 1)]

        mask = (X['local_date'].dt.date>train_start_idx) & (X['local_date'].dt.date<=train_split) & (X['minute_of_day']>=model_params['hour_range_start']) & (X['minute_of_day']<model_params['hour_range_stop'])
        X_train = X.loc[mask]
        y_train = y.loc[mask]

        # X_train.to_csv('X_train.csv', index=True, header=True)
        # y_train.to_csv('y_train.csv', index=True, header=True)

        X_train, y_train = remove_outliers(X_train, y_train, params, threshold=1.005)
        X_train = drop_ohlc_columns(X_train, list_X)

        registered_model_name = f'{params["model_type"]}_v{params["version"]}_{train_split.strftime("%Y%m%d")}'


        model, scaler = run_backtest_strategy(params, X_train, y_train, None, None, None, model_params)

        print("Saving model parameters to json...")
        model_params_path = os.path.join(params['models_path'], f"{registered_model_name}_model_params.json")
        save_model_params(model_params=model_params, file_path=model_params_path, logger=None)
        mlflow.log_artifact(local_path=model_params_path, artifact_path='model_params')

        if scaler is not None:
            # 2. Save scaler locally and log it as a separate artifact
            scaler_name = f"{registered_model_name}_scaler.joblib"
            scaler_path = os.path.join(params['models_path'], scaler_name)
            dump(scaler, scaler_path)
            mlflow.log_artifact(local_path=scaler_path, artifact_path="preprocessing")
        else:
            scaler_name=None
            scaler_path=None
        mlflow.log_params({'scaler': scaler_name})


        # Define example input and infer the signature (schema)
        input_example = X_train.iloc[-5:] # Use the last 5 rows as an example
        # Run example predictions on data (last records of X_train) and print the results
        if scaler is not None:
            input_example_scaled = scaler.transform(input_example)
            y_example = model.predict(input_example_scaled)
        else:
            y_example = model.predict(input_example)
        print(f"Last training rows predictions: {y_example}")
        mlflow.log_param('example_predictions', y_example)


        if params['model_type'] in ['xgb_regression', 'xgb_classification']:
            # predictions = model.predict(DMatrix(input_example))
            # signature = infer_signature(input_example, predictions)
            
            mlflow.xgboost.log_model(
            xgb_model=model,
            name=registered_model_name,
            # signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
            metadata={
                'train_start': train_start_idx,
                'train_end': train_split,
            }, params={
                'num_class': 3,
                'device': 'gpu',
                'learning_rate': model_params['learning_rate'],
                'max_depth': model_params['max_depth'],
                'subsample': model_params['subsample'],
                'gamma': model_params['gamma'],
                'objective': 'multi:softprob',
                'eval_metric': ['mlogloss', 'merror'],
            },
            extra_files=[scaler_path, model_params_path] if scaler_path is not None else [model_params_path]
            )


            importance_scores = model.get_score(importance_type='weight')
            sorted_by_values = dict(sorted(importance_scores.items(), key=lambda item: item[1]))
            with open(os.path.join(params['models_path'], 'importance_scores_weight.json'), "w") as file_name:
                json.dump(sorted_by_values, file_name)
            mlflow.log_artifact(os.path.join(params['models_path'], 'importance_scores_weight.json'), artifact_path='importance_scores')

            importance_scores = model.get_score(importance_type='gain')
            sorted_by_values = dict(sorted(importance_scores.items(), key=lambda item: item[1]))
            with open(os.path.join(params['models_path'], 'importance_scores_gain.json'), "w") as file_name:
                json.dump(sorted_by_values, file_name)
            mlflow.log_artifact(os.path.join(params['models_path'], 'importance_scores_gain.json'), artifact_path='importance_scores')

        elif params['model_type'] in ['linear_regression', 'logistic_regression', 'ridge_regression', 'lasso_regression', 'elasticnet_regression']:
            mlflow.sklearn.log_model(
            model=model,
            name=registered_model_name,
            input_example=input_example,
            registered_model_name=registered_model_name,
            metadata={
                'train_start': train_start_idx,
                'train_end': train_split,
            },
            extra_files=[scaler_path, model_params_path] if scaler_path is not None else [model_params_path]
            )

        elif params['model_type'] in ['svr_regression', 'svc_classification']:
            mlflow.sklearn.log_model(
                sk_model=model,
                name=registered_model_name,
                input_example=input_example,
                registered_model_name=registered_model_name,
                metadata={
                    'train_start': train_start_idx,
                    'train_end': train_split,
                    'scaler': scaler_name,
                }, params={
                    'kernel': 'rbf',
                    'C': model_params['C'],
                    'gamma': 'scale',
                    'epsilon': model_params['epsilon'],
                },
                extra_files=[scaler_path, model_params_path] if scaler_path is not None else [model_params_path]
            )

        else:
            raise ValueError(f"Invalid model type: {params['model_type']}")


    return registered_model_name
