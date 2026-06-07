from mlflow.entities import Experiment
import numpy as np
from datetime import date
import multiprocessing as mp
from random import random
from datetime import datetime

from src.data_utils.utils import getXy
from src.data_utils.features import build_target
from src.backtesting.strategies import do_backtest_Strategy2
import mlflow
# LOG_SPLITS_TABLE={}
# for num_splits in range(5,16):
#     LOG_SPLITS_TABLE[num_splits] = num_splits-1-(np.round(np.logspace(0,10,num=num_splits-1,base=0.93308)-0.5,3))*((num_splits-2)*2)

LOG_SPLITS_TABLE = {
    5: [1.0, 2.236, 3.22 , 4.   ],
    6: [1.0, 2.272, 3.344, 4.24 , 5.   ],
    7: [1.0 , 2.29, 3.42, 4.4 , 5.25, 6.  ],
    8: [1.0, 2.308, 3.472, 4.516, 5.44 , 6.268, 7.   ],
    9: [1.0, 2.316, 3.52 , 4.598, 5.578, 6.46 , 7.272, 8.   ],
    10: [1.0 , 2.328, 3.544, 4.664, 5.688, 6.616, 7.48 , 8.28 , 9.   ],
    11: [1.0,  2.332,  3.574,  4.708,  5.77 ,  6.742,  7.66 ,  8.506, 9.28 , 10.   ],
    12: [1.0,  2.34,  3.58,  4.76,  5.84,  6.86,  7.8 ,  8.68,  9.5 , 10.28, 11.  ],
    13: [1.0 ,  2.342,  3.596,  4.784,  5.906,  6.94 ,  7.93 ,  8.832, 9.712, 10.526, 11.274, 12.   ],
    14: [1.0 ,  2.344,  3.616,  4.816,  5.944,  7.024,  8.032,  8.968, 9.88 , 10.72 , 11.536, 12.28 , 13.   ],
    15: [1.0 ,  2.352,  3.626,  4.848,  5.992,  7.084,  8.124,  9.086, 10.022, 10.906, 11.738, 12.518, 13.272, 14.   ]
    }


def objective(trial, data: dict, index_base: int, indexes_higher: list, timeframes: list, timeframe_scalers: list, list_X: list, col_y: list, cutoff_date: date, unique_dates: list, mondays_indexes: list, splits_all: list, experiment_id: str) -> float:

    model_params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 400),
        'max_depth': trial.suggest_int('max_depth', 7, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.02),
        'subsample': trial.suggest_float('subsample', 0.95, 0.95),
        'gamma':  trial.suggest_float('gamma', 0.95, 0.95),
        # 'feature_fraction':  trial.suggest_float('feature_fraction', 0.9, 1),
        # 'num_leaves':  trial.suggest_int('num_leaves', 10, 200),

        'sma1_period': trial.suggest_int('sma1_period', 2, 20),
        'sma2_period': trial.suggest_int('sma2_period', 30, 90), 
        'bb_periods': trial.suggest_int('bb_periods', 15, 50),
        'bb_nbdev': trial.suggest_float('bb_nbdev', 2, 2.5),
        'ema1_period': trial.suggest_int('ema1_period', 5, 10),
        'ema2_period': trial.suggest_int('ema2_period', 10, 40),
        'sar_acc': trial.suggest_float('sar_acc', 0.01, 0.5), 
        'sar_max': trial.suggest_float('sar_max', 0.1, 1.5), 
        'midprice_window': trial.suggest_int('midprice_window', 2, 2), # 2,30
        'l1_fast': trial.suggest_int('l1_fast', 5, 20), # 15,3,10
        'l2_fast': trial.suggest_int('l2_fast', 2, 6), 
        'l3_fast': trial.suggest_int('l3_fast', 5, 15), 
        'l1_slow': trial.suggest_int('l1_slow', 10, 40), 
        'l2_slow': trial.suggest_int('l2_slow', 2, 6),
        'l3_slow': trial.suggest_int('l3_slow', 15, 25),
        'kama_trend_period': trial.suggest_int('kama_trend_period', 20, 40),

        'ha_candle_period': trial.suggest_int('ha_candle_period', 4, 30), 
        'dc_market_regime_period': trial.suggest_int('dc_market_regime_period', 4, 30), 
        'displacement_strength_period': trial.suggest_int('displacement_strength_period', 2, 40), 
        'displacement_strength': trial.suggest_float('displacement_strength', 1, 2),
        'displacement_hull_period': trial.suggest_int('displacement_hull_period', 10, 60), 
        #    'displacement_sma_period': trial.suggest_int('displacement_sma_period', 2, 30), 
        'displacement_hull_slope_period': trial.suggest_int('displacement_hull_slope_period', 5, 20),

        'gap_lookback': trial.suggest_int('gap_lookback', 3, 9),
        'gap_hull_period': trial.suggest_int('gap_hull_period', 4, 40),
        'gap_hull_slope_period': trial.suggest_int('gap_hull_slope_period', 2, 15),

        'market_regime_threshold': trial.suggest_float('market_regime_threshold', 0.001, 0.004),
        'tenkan_window': trial.suggest_int('tenkan_window', 4, 15), 
        'kijun_window': trial.suggest_int('kijun_window', 10, 80), 
        'cci_timeperiods': trial.suggest_int('cci_timeperiods', 3, 30),
        'macd_fastperiod': trial.suggest_int('macd_fastperiod', 6, 18), 
        'macd_slowperiod': trial.suggest_int('macd_slowperiod', 10, 40), 
        'macd_signalperiod': trial.suggest_int('macd_signalperiod', 3, 15),
        'price_distribution_window_size': trial.suggest_int('price_distribution_window_size', 5, 5),   # 5,50
        'price_distribution_percentile_threshold': trial.suggest_float('price_distribution_percentile_threshold', 0.2, 0.2), # 0.2,0.5
        'rsi_period': trial.suggest_int('rsi_period', 5, 40),
        'rsi_slope_period': trial.suggest_int('rsi_slope_period', 3, 20),
        'stoch_fastk_period': trial.suggest_int('stoch_fastk_period', 2, 15),
        'stoch_slowk_period': trial.suggest_int('stoch_slowk_period', 2, 15),
        'stoch_slowd_period': trial.suggest_int('stoch_slowd_period', 10, 30),
        'ppo_fastperiod': trial.suggest_int('ppo_fastperiod', 3, 15),
        'ppo_slowperiod': trial.suggest_int('ppo_slowperiod', 25, 45),

        'stochrsi_timeperiod': trial.suggest_int('stochrsi_timeperiod', 7, 15),
        'stochrsi_fastk_period': trial.suggest_int('stochrsi_fastk_period', 5, 25),
        'stochrsi_fastd_period': trial.suggest_int('stochrsi_fastd_period', 3, 20),
        'train_range_len': trial.suggest_int('train_range_len', 5, 5),  #10,15
        'test_range_len': trial.suggest_int('test_range_len', 3, 5),  #3,5
        'hour_range_start': trial.suggest_int('hour_range_start', 10, 11),
        'hour_range_stop': trial.suggest_int('hour_range_stop', 20, 20),
        'adx_timeperiod': trial.suggest_int('adx_timeperiod', 5, 5),      #5,15
        'di_timeperiod': trial.suggest_int('di_timeperiod', 5, 20),
        'macd_slope_period': trial.suggest_int('macd_slope_period', 9, 9),
        'sl': trial.suggest_float('sl', 0.002, 0.004),
        'tp': trial.suggest_float('tp', 0.002, 0.005),

        'atr_period': trial.suggest_int('atr_period', 3, 3),

        'stochrsik_slope_period': trial.suggest_int('stochrsik_slope_period', 3, 10),
        'stochk_slope_period': trial.suggest_int('stochk_slope_period', 3, 10),
        'willr_timeperiod': trial.suggest_int('willr_timeperiod', 10, 30),

        'ha_sign_ma_period': trial.suggest_int('ha_sign_ma_period', 4, 15),

        'target_tp': trial.suggest_float('target_tp', 0.002, 0.005),
        'ema_period': trial.suggest_int('ema_period', 15, 20),
        'ema_reversed_period': trial.suggest_int('ema_reversed_period', 5, 10),
        'threshold_long': trial.suggest_float('threshold_long', 0.8, 0.85),
        'threshold_short': trial.suggest_float('threshold_short', 0.15, 0.2),
        'pred_ewm_span': trial.suggest_int('pred_ewm_span', 1, 5),
    }

    data[index_base].loc[:,"labeling_binary"], data[index_base].loc[:,"labeling_dual_ema"], data[index_base].loc[:,"labeling_multi"] = build_target(data[index_base], \
        open_col='Open', high_col='High', low_col='Low', high_time_col="high_time", low_time_col="low_time", \
        tp=model_params['target_tp'], ema_period=model_params['ema_period'], ema_reversed_period=model_params['ema_reversed_period'], \
        threshold_long=model_params['threshold_long'], threshold_short=model_params['threshold_short'])

    p={}
    p[7] = model_params
    p[10] = model_params
    X, y, X_columns = getXy(data, index_base, indexes_higher, model_params, p, timeframes, timeframe_scalers, list_X, col_y[0], cutoff_date, col_open="Open", col_high="High", col_low="Low", col_close="Close")
    y=y+1

#    X = X.loc[(X.index.hour>=model_params['hour_range_start']) & (X.index.hour<=model_params['hour_range_stop'])]

    #print(lag_f.get_feature_names_out())

    #ml_data.to_csv('ml_data1.csv')
    num_splits = 11

    # Split the dataset to test and train sets
    # Split the initial 70% of the data as training set and the remaining 30% data as the testing set
    num_mondays = len(mondays_indexes)
    splits_array = LOG_SPLITS_TABLE[num_splits]
    mondays_splits = [int(2 + (num_mondays-2) * (random()/2 + i) / num_splits) for i in splits_array] # range(1, num_splits)]
    train_splits = [mondays_indexes[i] for i in mondays_splits]
    print(train_splits)
    scores = []
    stats = []
    sharpe = []
    sortino = []
    calmar = []
    splits_all.append(train_splits)
    total_score = 0
    start_time = datetime.now()
    print(start_time.strftime('%H:%M:%S'))

    with mp.Pool(10) as p:
        tuples = []
        for idx,i in enumerate(train_splits):

            train_split = unique_dates[i]
            train_start_idx = unique_dates[max(i-model_params['train_range_len']*5, 1)]
            test_end_idx = unique_dates[min(i+model_params['test_range_len']*5, len(unique_dates)-5)]

            X_train, X_test = X.loc[(X.index.date>train_start_idx) & (X.index.date<=train_split)], X.loc[(X.index.date>train_split) & (X.index.date<=test_end_idx)]
            y_train, y_test = y.loc[(X.index.date>train_start_idx) & (X.index.date<=train_split)], y.loc[(X.index.date>train_split) & (X.index.date<=test_end_idx)]
            print(train_start_idx, train_split, test_end_idx)
            data_target = X_test.loc[:,['Open','High','Low','Close']]
            data_target['sl']=model_params['sl']
            data_target['tp']=model_params['tp']
            data_target['DaytradingExit'] = ((data_target.index.date != data_target.index.to_series().shift(periods=-1).dt.date) | (data_target.index.date != data_target.index.to_series().shift(periods=-2).dt.date))


#            data_target = data_target.join(ml_data['atr']).ffill().bfill()
            tuples.append((X_train, X_test, y_train, y_test, data_target, model_params, 1))         # exponential_growth(1, 0.02, num_splits-idx-1)

        results = p.starmap(do_backtest_Strategy2, tuples)
    
    results.sort(key=lambda res: res[5]['Start'])

    for res in results:
        scores.append(res[0])
        sharpe.append(res[1])
        sortino.append(res[2])
        calmar.append(res[3])
        stats.append(res[5])

#        y_pred_all = y_pred_all.combine_first(res[4])
#    y_pred_all.to_csv('y_pred_all_opt1.csv')

    total_score = sum(scores)
    scores_std=np.std(scores)

    print(f'Splits: {train_splits}')
    print(f'Profits: {scores} Sum: {total_score} Stddev: {scores_std}')

    print(f'Sharpe: {sharpe} Avg: {np.mean(sharpe)}')
    print(f'Sortino: {sortino} Avg: {np.mean(sortino)}')
    print(f'Calmar: {calmar} Avg: {np.mean(calmar)}')


    # Set tracking URI
    print("Setting MLFlow tracking and experiment ID for active run...")
    mlflow.set_tracking_uri("http://localhost:5000/")
    mlflow.set_experiment(experiment_id=experiment_id)
    # print(f"Experiment ID: {experiment_id}")


    for idx, res in enumerate(results):
        with mlflow.start_run(nested=True, run_name=f"Child_Run_{idx}") as child_run:

            try:
                # mlflow.log_metric('Start', int(res[5]['Start'].timestamp()))
                mlflow.log_metric('Start', res[5]['Start'].year*10000+res[5]['Start'].month*100+res[5]['Start'].day)
                # mlflow.log_metric('End', int(res[5]['End'].timestamp()))
                mlflow.log_metric('End', res[5]['End'].year*10000+res[5]['End'].month*100+res[5]['End'].day)
                mlflow.log_metric('Equity_Final', res[5]['Equity Final [$]'])
                mlflow.log_metric('Equity_Peak', res[5]['Equity Peak [$]'])
                mlflow.log_metric('Commissions', res[5]['Commissions [$]'])
                mlflow.log_metric('Return_Percentage', res[5]['Return [%]'])
                mlflow.log_metric('Buy_Hold_Return_Percentage', res[5]['Buy & Hold Return [%]'])
                mlflow.log_metric('Sharpe_Ratio', res[5]['Sharpe Ratio'])
                mlflow.log_metric('Sortino_Ratio', res[5]['Sortino Ratio'])
                mlflow.log_metric('Calmar_Ratio', res[5]['Calmar Ratio'])
                mlflow.log_metric('Max_Drawdown_Percentage', res[5]['Max. Drawdown [%]'])
                mlflow.log_metric('Avg_Drawdown_Percentage', res[5]['Avg. Drawdown [%]'])
                mlflow.log_metric('Max_Drawdown_Duration_Seconds', res[5]['Max. Drawdown Duration'].seconds)
                mlflow.log_metric('Avg_Drawdown_Duration_Seconds', res[5]['Avg. Drawdown Duration'].seconds)
                mlflow.log_metric('Num_Trades', res[5]['# Trades'])
                mlflow.log_metric('Win_Rate_Percentage', res[5]['Win Rate [%]'])
                mlflow.log_metric('Best_Trade_Percentage', res[5]['Best Trade [%]'])
                mlflow.log_metric('Worst_Trade_Percentage', res[5]['Worst Trade [%]'])
                mlflow.log_metric('Avg_Trade_Percentage', res[5]['Avg. Trade [%]'])
                mlflow.log_metric('Max_Trade_Duration_Seconds', res[5]['Max. Trade Duration'].seconds)
                mlflow.log_metric('Avg_Trade_Duration_Seconds', res[5]['Avg. Trade Duration'].seconds)
                mlflow.log_metric('Profit_Factor', res[5]['Profit Factor'])
                mlflow.log_metric('Expectancy_Percentage', res[5]['Expectancy [%]'])
                mlflow.log_metric('SQN', res[5]['SQN'])
                mlflow.log_metric('Kelly_Criterion', res[5]['Kelly Criterion'])
                print(f"Child run {idx} logging ended")
            except Exception as e:
                # mlflow.log_params(res[5])
                print(res[5])
                print(f"Error logging child run {idx}: {e}")

    return total_score + np.mean(np.sort(scores)[:3]) #* np.sqrt(max(np.mean(sortino)+np.mean(calmar), 1)) / (parameters['hour_range_stop']-parameters['hour_range_start']+2)



# def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
#     """Save the model run ID and path to a JSON file."""
#     try:
#         # Create a dictionary with the info you want to save
#         model_info = {
#             'run_id': run_id,
#             'model_path': model_path
#         }
#         # Save the dictionary as a JSON file
#         with open(file_path, 'w') as file:
#             json.dump(model_info, file, indent=4)
#         logger.debug('Model info saved to %s', file_path)
#     except Exception as e:
#         logger.error('Error occurred while saving the model info: %s', e)
#         raise
