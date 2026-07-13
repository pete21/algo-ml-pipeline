import json
import numpy as np
from datetime import date
import multiprocessing as mp
from random import random
from datetime import datetime
import os

# from xgboost import DMatrix
from src.data_utils.utils import getXy
from src.data_utils.features import build_target
from src.backtesting.strategies import do_backtest_Strategy2_trading, do_backtest_Strategy2_evals, do_backtest_Strategy2_training
import mlflow

from src.model.mlflow_utils import save_model_params
# LOG_SPLITS_TABLE={}
# for num_splits in range(5,16):
#     LOG_SPLITS_TABLE[num_splits] = num_splits-1-(np.round(np.logspace(0,10,num=num_splits-1,base=0.93308)-0.5,3))*((num_splits-2)*2)

LOG_SPLITS_TABLE = {
    # 2: [1.0],
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
    16: [ 1.0, 2.344,  3.632,  4.864,  6.04 ,  7.132,  8.196,  9.204, 10.156, 11.052, 11.92 , 12.76 , 13.544, 14.272, 15.   ],
    17: [ 1.0, 2.35,  3.64,  4.87,  6.07,  7.18,  8.26,  9.28, 10.27, 11.2 , 12.1 , 12.94, 13.75, 14.53, 15.28, 16.  ],
    18: [ 1.0, 2.344,  3.656,  4.904,  6.088,  7.24 ,  8.328,  9.352, 10.376, 11.336, 12.232, 13.128, 13.96 , 14.76 , 15.56 , 16.296, 17.   ],
    19: [ 1.0, 2.36,  3.652,  4.91 ,  6.1  ,  7.256,  8.378,  9.432, 10.452, 11.438, 12.39 , 13.274, 14.158, 14.974, 15.79 , 16.538, 17.286, 18.   ],
    20: [ 1.0, 2.368,  3.664,  4.924,  6.148,  7.3  ,  8.416,  9.496, 10.54 , 11.548, 12.484, 13.42 , 14.32 , 15.184, 16.012, 16.804, 17.56 , 18.28 , 19.   ],
    21: [ 1.0, 2.368,  3.66 ,  4.952,  6.168,  7.346,  8.448,  9.55 , 10.614, 11.64 , 12.59 , 13.54 , 14.452, 15.326, 16.2  , 16.998, 17.796, 18.556, 19.278, 20.   ],
    22: [ 1.0, 2.36,   3.68 ,  4.96 ,  6.16 ,  7.36 ,  8.52 ,  9.6 ,  10.68,  11.72,  12.72,  13.68,  14.6 ,  15.52,  16.36,  17.2 ,  18.  ,  18.8 ,  19.56,  20.28, 21.  ],
    23: [ 1.0, 2.344,  3.688,  4.948,  6.208,  7.384,  8.56,   9.652, 10.744, 11.794, 12.802, 13.768, 14.734, 15.658, 16.54,  17.38,  18.22,  19.018, 19.816, 20.572, 21.286, 22.  ],
    24: [ 1.0, 2.364,  3.684,  4.96,   6.192,  7.424,  8.568,  9.712, 10.812, 11.868, 12.88,  13.892, 14.86,  15.784, 16.664, 17.544, 18.424, 19.216, 20.052, 20.8,   21.548, 22.296, 23.  ],
    25: [ 1.0, 2.38,   3.668,  4.956,  6.198,  7.44,   8.59,   9.74,  10.844, 11.902, 12.96,  13.972, 14.938, 15.904, 16.824, 17.698, 18.572, 19.446, 20.228, 21.056, 21.792, 22.574, 23.264, 24.  ],
    }


def objective(trial, data: dict, params: dict, cutoff_date: date, unique_dates: list, experiment_id: str, model_params_override: dict = None) -> float:

    index_base = params['index_base']
    indexes_higher = params['indexes_higher']
    timeframes = params['timeframes']
    timeframe_scalers = params['timeframe_scalers']
    list_X = params['list_X']
    col_y = params['col_y']
    lags = params['lags']
    
    print("Trial: ", trial.number if trial else "Evaluation")
    run_name = f"Trial_{trial.number}" if trial else "Evaluation"
    # Set tracking URI
    print("Setting MLFlow tracking and experiment ID...")
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    mlflow.set_experiment(experiment_id=experiment_id)
    # print(f"Experiment ID: {experiment_id}")

    # mlflow.xgboost.autolog()

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param('cutoff_date', cutoff_date)
        if params['evals_strategy']:
            mlflow.log_param('_strategy', 'do_backtest_Strategy2_evals')
        else:
            mlflow.log_param('_strategy', 'do_backtest_Strategy2_trading')
        mlflow.log_params(params)

        model_params = model_params_override or {
            'n_estimators': trial.suggest_int('n_estimators', 380, 400, step=5),
            'max_depth': trial.suggest_int('max_depth', 7, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.02, step=0.001),
            'subsample': trial.suggest_float('subsample', 0.95, 0.95),
            'gamma':  trial.suggest_float('gamma', 0.95, 0.95),
            # 'feature_fraction':  trial.suggest_float('feature_fraction', 0.9, 1),
            # 'num_leaves':  trial.suggest_int('num_leaves', 10, 200),

            'sma1_period': trial.suggest_int('sma1_period', 7, 15),
            'sma2_period': trial.suggest_int('sma2_period', 70, 100), 
            'bb_periods': trial.suggest_int('bb_periods', 30, 55),
            'bb_nbdev': trial.suggest_float('bb_nbdev', 2, 2.25),
            'ema1_period': trial.suggest_int('ema1_period', 5, 10),
            'ema2_period': trial.suggest_int('ema2_period', 15, 35),
            'sar_acc': trial.suggest_float('sar_acc', 0.3, 0.6), 
            'sar_max': trial.suggest_float('sar_max', 0.4, 1), 
            'midprice_window': trial.suggest_int('midprice_window', 2, 2), # 2,30
            'l1_fast': trial.suggest_int('l1_fast', 4, 10), # 15,3,10
            'l2_fast': trial.suggest_int('l2_fast', 3, 5), 
            'l3_fast': trial.suggest_int('l3_fast', 10, 20), 
            'l1_slow': trial.suggest_int('l1_slow', 25, 40), 
            'l2_slow': trial.suggest_int('l2_slow', 5, 9),
            'l3_slow': trial.suggest_int('l3_slow', 20, 30),
            'kama_trend_period': trial.suggest_int('kama_trend_period', 20, 35),

            'ha_candle_period': trial.suggest_int('ha_candle_period', 20, 40), 
            'dc_market_regime_period': trial.suggest_int('dc_market_regime_period', 20, 35), 
            'displacement_strength_period': trial.suggest_int('displacement_strength_period', 20, 35), 
            'displacement_strength': trial.suggest_float('displacement_strength', 1.2, 1.8),
            'displacement_hull_period': trial.suggest_int('displacement_hull_period', 10, 50), 
            #    'displacement_sma_period': trial.suggest_int('displacement_sma_period', 2, 30), 
            'displacement_hull_slope_period': trial.suggest_int('displacement_hull_slope_period', 5, 10),

            'gap_lookback': trial.suggest_int('gap_lookback', 2, 7),
            'gap_hull_period': trial.suggest_int('gap_hull_period', 8, 15),             # minimum 4
            'gap_hull_slope_period': trial.suggest_int('gap_hull_slope_period', 6, 15),

            'market_regime_threshold': trial.suggest_float('market_regime_threshold', 0.003, 0.005),
            'tenkan_window': trial.suggest_int('tenkan_window', 6, 12), 
            'kijun_window': trial.suggest_int('kijun_window', 45, 75), 
            'cci_timeperiods': trial.suggest_int('cci_timeperiods', 20, 35),
            'macd_fastperiod': trial.suggest_int('macd_fastperiod', 5, 15), 
            'macd_slowperiod': trial.suggest_int('macd_slowperiod', 30, 40), 
            'macd_signalperiod': trial.suggest_int('macd_signalperiod', 5, 10),
            'price_distribution_window_size': trial.suggest_int('price_distribution_window_size', 5, 5),   # 5,50
            'price_distribution_percentile_threshold': trial.suggest_float('price_distribution_percentile_threshold', 0.2, 0.2), # 0.2,0.5
            'rsi_period': trial.suggest_int('rsi_period', 7, 21),
            'rsi_slope_period': trial.suggest_int('rsi_slope_period', 12, 20),
            'stoch_fastk_period': trial.suggest_int('stoch_fastk_period', 5, 12),
            'stoch_slowk_period': trial.suggest_int('stoch_slowk_period', 5, 15),
            'stoch_slowd_period': trial.suggest_int('stoch_slowd_period', 20, 28),
            'ppo_fastperiod': trial.suggest_int('ppo_fastperiod', 8, 15),
            'ppo_slowperiod': trial.suggest_int('ppo_slowperiod', 30, 45),

            'stochrsi_timeperiod': trial.suggest_int('stochrsi_timeperiod', 12, 18),
            'stochrsi_fastk_period': trial.suggest_int('stochrsi_fastk_period', 3, 6),
            'stochrsi_fastd_period': trial.suggest_int('stochrsi_fastd_period', 8, 15),
            'train_range_len': trial.suggest_int('train_range_len', 15, 20),
            'test_range_len': trial.suggest_int('test_range_len', 4, 4),  #3,5
            'hour_range_start': trial.suggest_int('hour_range_start', 7*60, 10*60, step=15),
            # 'hour_range_stop': trial.suggest_int('hour_range_stop', 20, 20),
            'adx_timeperiod': trial.suggest_int('adx_timeperiod', 5, 5),      #5,15
            'di_timeperiod': trial.suggest_int('di_timeperiod', 5, 15),
            'macd_slope_period': trial.suggest_int('macd_slope_period', 9, 9),
            # 'sl': trial.suggest_float('sl', 0.003, 0.004) if not params['evals_strategy'] else 0,
            'tp': trial.suggest_float('tp', 0.0025, 0.0035) if not params['evals_strategy'] else trial.suggest_int('tp', 50, 150),

            'atr_period': trial.suggest_int('atr_period', 4, 8),

            'stochrsik_slope_period': trial.suggest_int('stochrsik_slope_period', 10, 15),
            'stochk_slope_period': trial.suggest_int('stochk_slope_period', 10, 16),
            'willr_timeperiod': trial.suggest_int('willr_timeperiod', 25, 35),

            'ha_sign_ma_period': trial.suggest_int('ha_sign_ma_period', 7, 12),

            'target_tp': trial.suggest_float('target_tp', 0.0025, 0.003),
            'ema_period': trial.suggest_int('ema_period', 20, 30),
            'ema_reversed_period': trial.suggest_int('ema_reversed_period', 6, 10),
            'threshold_long': trial.suggest_float('threshold_long', 0.81, 0.84),
            'threshold_short': trial.suggest_float('threshold_short', 0.16, 0.2),
            'pred_ewm_span': trial.suggest_float('pred_ewm_span', 1.2, 1.8, step=0.1),
            'pca_ichimoku': trial.suggest_categorical('pca_ichimoku', [False]),
            'pca_kama': trial.suggest_categorical('pca_kama', [False]),
            'weekday': trial.suggest_categorical('weekday', [2]),                     # 0: Monday, 2: Wednesday, 4: Friday
        }

        if not model_params_override:
            model_params['hour_range_stop'] = trial.suggest_int('hour_range_stop', model_params['hour_range_start'] + 5*60, model_params['hour_range_start'] + 5*60)

            if params['evals_strategy']:
                model_params['sl'] = trial.suggest_int('sl', model_params['tp'] // 1.5, model_params['tp'] // 1.5)
            else:
                model_params['sl'] = trial.suggest_int('sl', model_params['tp'], model_params['tp'])


        data[index_base].loc[:,"labeling_binary"], data[index_base].loc[:,"labeling_dual_ema"], data[index_base].loc[:,"labeling_multi"] = build_target(data[index_base], \
            open_col='Open', high_col='High', low_col='Low', high_time_col="high_time", low_time_col="low_time", \
            tp=model_params['target_tp'], ema_period=model_params['ema_period'], ema_reversed_period=model_params['ema_reversed_period'], \
            threshold_long=model_params['threshold_long'], threshold_short=model_params['threshold_short'])

        p={}
        for i in indexes_higher:
            p[i] = model_params
        X, y, X_columns = getXy(data, index_base, indexes_higher, model_params, p, timeframes, timeframe_scalers, list_X, col_y[0], cutoff_date, lags, col_open="Open", col_high="High", col_low="Low", col_close="Close")
        y=y+1
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
        mlflow.log_param('X_columns', X_columns)
        mlflow.log_param('y_col', col_y[0])
        mlflow.log_param('index_base', index_base)
        mlflow.log_param('indexes_higher', indexes_higher)

        tuples = []
        for i in train_splits:

            train_split = unique_dates[i]
            train_start_idx = unique_dates[max(i-model_params['train_range_len']*5, 1)]
            test_end_idx = unique_dates[min(i+model_params['test_range_len']*5, len(unique_dates)-5)]
            print(train_start_idx, train_split, test_end_idx)

            X_train = X.loc[(X['local_date'].dt.date>train_start_idx) & (X['local_date'].dt.date<=train_split) & (X['minute_of_day']>=model_params['hour_range_start']) & (X['minute_of_day']<model_params['hour_range_stop'])]
            y_train = y.loc[(X['local_date'].dt.date>train_start_idx) & (X['local_date'].dt.date<=train_split) & (X['minute_of_day']>=model_params['hour_range_start']) & (X['minute_of_day']<model_params['hour_range_stop'])]

            X_test = X.loc[(X['local_date'].dt.date>train_split) & (X['local_date'].dt.date<=test_end_idx)]
            # y_test = y.loc[(X['local_date'].dt.date>train_split) & (X['local_date'].dt.date<=test_end_idx)]
            y_test = None

            data_target = X_test.loc[:,['Open','High','Low','Close','minute_of_day']]
            data_target['sl']=model_params['sl']
            data_target['tp']=model_params['tp']
            
            # data_target['DaytradingExit'] = ((data_target.index.date != data_target.index.to_series().shift(periods=-1).dt.date) | (data_target.index.date != data_target.index.to_series().shift(periods=-2).dt.date))
            data_target['DaytradingExit'] = (data_target['minute_of_day'] >= 21*60-15) & (data_target['minute_of_day'] <= 21*60)

            X_test = X_test.loc[(X_test['minute_of_day']>=model_params['hour_range_start']) & (X_test['minute_of_day']<model_params['hour_range_stop'])]

            X_train=X_train.drop(columns=['minute_of_day', 'local_date'])
            X_test=X_test.drop(columns=['minute_of_day', 'local_date'])
            # X_train.to_csv('X_train_'+str(i)+'.csv')
            # y_train.to_csv('y_train_'+str(i)+'.csv')
            #data_target = data_target.join(ml_data['atr']).ffill().bfill()
            tuples.append((X_train, X_test, y_train, y_test, data_target, model_params, 1))         # exponential_growth(1, 0.02, num_splits-idx-1)

        results = []
        # with mp.Pool(10) as p:
        with mp.Pool(12, maxtasksperchild=1) as p:
            if params['evals_strategy']:
                results = p.starmap(do_backtest_Strategy2_evals, tuples)
            else:
                results = p.starmap(do_backtest_Strategy2_trading, tuples)
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
            calmar = [s['Calmar Ratio'] for s in stats]
            sharpe_mean = np.nanmean(sharpe)
            sortino_mean = np.nanmean(sortino)
            calmar_mean = np.nanmean(calmar)

            print(f'Sharpe: {sharpe} Avg: {sharpe_mean}')
            print(f'Sortino: {sortino} Avg: {sortino_mean}')
            print(f'Calmar: {calmar} Avg: {calmar_mean}')

            mlflow.log_metric('sharpe_ratio', sharpe_mean)
            mlflow.log_metric('sortino_mean', sortino_mean)
            mlflow.log_metric('calmar_mean', calmar_mean)

            total_return_percentage = total_profit/1000
            mlflow.log_metric('total_return_percentage', total_return_percentage)
            mlflow.log_metric('total_expectancy_percentage', total_return_percentage / total_trades)

            wins = [ s['_trades'].loc[s['_trades']['PnL']>0,'PnL'] for s in stats ]
            losses = [ s['_trades'].loc[s['_trades']['PnL']<0,'PnL'] for s in stats ]
            draws = [ s['_trades'].loc[s['_trades']['PnL']==0,'PnL'] for s in stats ]
            
            wins_value = sum([w.sum(skipna=True) for w in wins])
            losses_value = sum([l.sum(skipna=True) for l in losses])
            win_trades = sum([w.count() for w in wins])
            loss_trades = sum([l.count() for l in losses])
            mlflow.log_metric('win_trades', win_trades)
            mlflow.log_metric('loss_trades', loss_trades)
            mlflow.log_metric('draw_trades', sum([d.count() for d in draws]))
            
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
                optimisation_score = (total_profit + np.nanmean(np.sort(profits)[:int(num_splits/4)])) * (0.5 + sharpe_coeff/10 + win_rate + np.log(total_trades)/20 - avg_win/avg_loss/10)
            else:
                optimisation_score = (total_profit + np.nanmean(np.sort(profits)[:int(num_splits/4)])) * (1 + sharpe_coeff/10 + win_rate/10 + np.log(total_trades)/10 - avg_win/avg_loss/20) #* np.sqrt(max(np.mean(sortino)+np.mean(calmar), 1)) / (parameters['hour_range_stop']-parameters['hour_range_start']+2)

            mlflow.log_metric('optimisation_score', optimisation_score)
            
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
                            mlflow.log_metric('Calmar_Ratio', res['Calmar Ratio'])
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

                            draws = res['_trades'].loc[res['_trades']['PnL']==0, 'PnL']
                            mlflow.log_metric('draw_trades', draws.count())

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

            # last_model_gpu = results[-1][2]
            # mlflow.xgboost.log_model(xgb_model=last_model_gpu, name='xgb_model_gpu',
            # # registered_model_name=f'xgb_model_gpu_{run_name}',
            # metadata={
            #     'train_split': train_splits[-1],
            #     'train_start': unique_dates[max(train_splits[-1]-model_params['train_range_len']*5, 1)],
            #     'train_end': unique_dates[train_splits[-1]],
            # }, params={
            #     'num_class': 3,
            #     'device': 'gpu',
            #     'learning_rate': model_params['learning_rate'],
            #     'max_depth': model_params['max_depth'],
            #     'subsample': model_params['subsample'],
            #     'gamma': model_params['gamma'],
            #     'objective': 'multi:softprob',
            #     'eval_metric': 'auc',
            # })

            # importance_scores = last_model_gpu.get_score(importance_type='weight')
            # sorted_by_values = dict(sorted(importance_scores.items(), key=lambda item: item[1]))
            # with open(os.path.join(params['models_path'], f'importance_scores_weight.json'), "w") as file_name:
            #     json.dump(sorted_by_values, file_name)
            # mlflow.log_artifact(os.path.join(params['models_path'], f'importance_scores_weight.json'), artifact_path='importance_scores')
            
        else:
            mlflow.log_metric('total_trades', 0)
            mlflow.log_metric('total_profit', 0)
            mlflow.log_metric('win_trades', 0)
            mlflow.log_metric('loss_trades', 0)
            mlflow.log_metric('optimisation_score', optimisation_score)


    return optimisation_score


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


        data[index_base].loc[:,"labeling_binary"], data[index_base].loc[:,"labeling_dual_ema"], data[index_base].loc[:,"labeling_multi"] = build_target(data[index_base], \
            open_col='Open', high_col='High', low_col='Low', high_time_col="high_time", low_time_col="low_time", \
            tp=model_params['target_tp'], ema_period=model_params['ema_period'], ema_reversed_period=model_params['ema_reversed_period'], \
            threshold_long=model_params['threshold_long'], threshold_short=model_params['threshold_short'])

        p={}
        for i in indexes_higher:
            p[i] = model_params
        X, y, X_columns = getXy(data, index_base, indexes_higher, model_params, p, timeframes, timeframe_scalers, list_X, col_y[0], date(2026,1,1), lags, col_open="Open", col_high="High", col_low="Low", col_close="Close")
        y=y+1

        train_splits = [train_split_index]
        print(train_splits)

        print(datetime.now().strftime('%H:%M:%S'))

        mlflow.log_params(model_params)
        mlflow.log_param('train_splits', train_splits)
        mlflow.log_param('X_columns', X_columns)
        mlflow.log_param('y_col', col_y[0])
        mlflow.log_param('index_base', index_base)
        mlflow.log_param('indexes_higher', indexes_higher)


        train_split = unique_weekdates[train_splits[-1]]
        train_start_idx = unique_weekdates[max(train_splits[-1]-model_params['train_range_len']*5, 1)]

        X_train = X.loc[(X['local_date'].dt.date>train_start_idx) & (X['local_date'].dt.date<=train_split)]
        y_train = y.loc[(X['local_date'].dt.date>train_start_idx) & (X['local_date'].dt.date<=train_split)]
        X_train=X_train.drop(columns=['minute_of_day', 'local_date'])

        model_gpu = do_backtest_Strategy2_training(X_train, y_train, model_params)

        # Define example input and infer the signature (schema)
        input_example = X_train.iloc[-5:] # Use the last 5 rows as an example
        # predictions = model_gpu.predict(DMatrix(input_example))
        # signature = infer_signature(input_example, predictions)
        
        registered_model_name = f'{os.getenv("MODEL_NAME")}_{train_split.strftime("%Y%m%d")}'
        mlflow.xgboost.log_model(xgb_model=model_gpu, name=os.getenv('MODEL_NAME'),
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
            'eval_metric': 'auc',
        })

        importance_scores = model_gpu.get_score(importance_type='weight')
        sorted_by_values = dict(sorted(importance_scores.items(), key=lambda item: item[1]))
        with open(os.path.join(params['models_path'], 'importance_scores_weight.json'), "w") as file_name:
            json.dump(sorted_by_values, file_name)
        mlflow.log_artifact(os.path.join(params['models_path'], 'importance_scores_weight.json'), artifact_path='importance_scores')

        importance_scores = model_gpu.get_score(importance_type='gain')
        sorted_by_values = dict(sorted(importance_scores.items(), key=lambda item: item[1]))
        with open(os.path.join(params['models_path'], 'importance_scores_gain.json'), "w") as file_name:
            json.dump(sorted_by_values, file_name)
        mlflow.log_artifact(os.path.join(params['models_path'], 'importance_scores_gain.json'), artifact_path='importance_scores')


        # Save the trained model in the root directory
        print("Saving model parameters to json...")
        model_params_path = os.path.join(params['models_path'], "model_params_registration.json")
        save_model_params(model_params=model_params, file_path=model_params_path, logger=None)
        mlflow.log_artifact(local_path=model_params_path, artifact_path='model_params')

    return registered_model_name
