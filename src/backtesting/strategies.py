import numpy as np
from backtesting import Backtest
from backtesting.lib import Strategy

# from xgboost import XGBClassifier
from xgboost import DMatrix, train
from sklearn.utils.class_weight import compute_sample_weight
from random import randint
import pandas as pd

# EMPTY_STATS = {
#     'Start': datetime.now(),
#     'End': None,
#     'Duration': None,
#     'Exposure Time [%]': None,
#     'Equity Final [$]': 100000,
#     'Equity Peak [$]': None,
#     'Commissions [$]': None,
#     'Return [%]': 0,
#     'Buy & Hold Return [%]': None,
#     'Return (Ann.) [%]': None,
#     'Volatility (Ann.) [%]': None,
#     'CAGR [%]': None,
#     'Sharpe Ratio': 0,
#     'Sortino Ratio': 0,
#     'Calmar Ratio': 0,
#     'Alpha [%]': None,
#     'Beta': None,
#     'Max. Drawdown [%]': 0,
#     'Avg. Drawdown [%]': 0,
#     'Max. Drawdown Duration': None,
#     'Avg. Drawdown Duration': None,
#     '# Trades': 0,
#     'Win Rate [%]': 0,
#     'Best Trade [%]': 0,
#     'Worst Trade [%]': 0,
#     'Avg. Trade [%]': 0,
#     'Max. Trade Duration': None,
#     'Avg. Trade Duration': None,
#     'Profit Factor': 0,
#     'Expectancy [%]': 0,
#     'SQN': None,
#     'Kelly Criterion': None,
#     '_strategy': '',
#     '_trades': [],
#     '_equity_curve': [],
# }


######################################################## DAYTRADING STRATEGY ########################################################

def do_backtest_Strategy2_trading(X_train, X_test, y_train, y_test, data_target, params, weighting):
    rand_int = randint(1000000, 2000000)
    params_gpu = {
        'num_class': 3, 'device': 'gpu',
        'learning_rate': params['learning_rate'],
        # 'n_estimators': params['n_estimators'],
        'max_depth': params['max_depth'],
        'subsample': params['subsample'],
        'gamma': params['gamma'],
        'objective': 'multi:softprob',
        'seed': rand_int,
        'eval_metric': 'auc', #'merror',
        # 'verbose_eval': 1000
    }

    sample_weights = compute_sample_weight(
        class_weight='balanced',
        y=y_train
    )
    dtrain_gpu = DMatrix(X_train, label=y_train, weight=sample_weights)
    dtest_gpu = DMatrix(X_test)
    # print("dtrain_gpu: ", dtrain_gpu.shape)
    # print("dtest_gpu: ", dtest_gpu.shape)
    # print("sample_weights: ", sample_weights.shape)
    # print("y_train: ", y_train.shape)
    # print("y_test: ", y_test.shape)
    # print("X_train: ", X_train.shape)
    # print("X_test: ", X_test.shape)
    # print("data_target: ", data_target.shape)
    # print("weighting: ", weighting)

    print("Training model...")
    model_gpu = train(params_gpu, dtrain_gpu, num_boost_round=params['n_estimators'])
    # importance_scores = model_gpu.get_score(importance_type='weight')
    # sorted_by_values = dict(sorted(importance_scores.items(), key=lambda item: item[1]))
    # json.dump( sorted_by_values, open( f'importance_scores_optim_{rand_int}.json', 'w' ) )

    # if X_test.count() < 5:
    #     return pd.Series(np.zeros(X_test.count()), index=X_test.index, name="y_pred"), EMPTY_STATS, model_gpu


    print("Predicting...")
    y_pred = model_gpu.predict(dtest_gpu)
    # print("X_test: ", X_test)
    # print("y_pred: ", y_pred)
    # print("Predicted.")
    # print("Creating y_series...")
    y_pred_expected = np.matmul(y_pred, np.array([[-1],[0],[1]]))
    # print("y_pred_expected: ", y_pred_expected)
    # y_series = pd.Series(y_pred-1, index=X_test.index, name="y_pred")
    # y_series = pd.Series(y_pred_expected.flatten(), index=X_test.index, name="y_pred").rolling(window=params['pred_avg_period'], min_periods=1).mean()
    y_series = pd.Series(y_pred_expected.flatten(), index=X_test.index, name="y_pred").ewm(span=params['pred_ewm_span'], adjust=False).mean()

    print("Joining y_series to data_target...")
    data_target = data_target.join(y_series, how='left')

    print("Backtesting...")
    # data_target.to_csv(f'data_target_optim_{rand_int}.csv')

    bt = Backtest(data_target,
        Strategy2_opt_daytrading,
        cash=100000,
        spread=0,
        commission=0.00008,
        margin=1,
        trade_on_close=False,
        hedging=False,
        exclusive_orders=True,
        finalize_trades=True)
    stats = bt.run()

    return y_series, stats #, model_gpu


# def do_backtest_Strategy2_trading(X_train, X_test, y_train, y_test, data_target, params, weighting):
#     rand_int = randint(1000000, 2000000)
#     model_xgb = XGBClassifier(num_class=3, device='gpu',
#                     learning_rate=params['learning_rate'],
#                     n_estimators=params['n_estimators'],
#                     max_depth=params['max_depth'],
#                     subsample=params['subsample'],
#                     gamma=params['gamma'],
#                     objective='multi:softprob',
#                     random_state=rand_int,
#                     # early_stopping_rounds=10,
#                     eval_metric='merror',
#                     # min_delta=0.01,
#                     # num_leaves=params['num_leaves'],
#                     # feature_fraction=params['feature_fraction']
#                     )                   # model

#     sample_weights = compute_sample_weight(
#         class_weight='balanced',
#         y=y_train
#     )

#     model_xgb.fit(X_train, y_train, sample_weight=sample_weights)                           # cp.array(X_train)
#     y_pred = model_xgb.predict(X_test)

#     y_series = pd.Series(y_pred-1, index=X_test.index, name="y_pred")

#     data_target = data_target.join(y_series)

# #    data_target.to_csv('data_target_optim_1.csv')

#     bt = Backtest(data_target,
#         Strategy2_opt_daytrading,
#         cash=100000,
#         spread=0,
#         commission=0.0001,
#         margin=1,
#         trade_on_close=False,
#         hedging=False,
#         exclusive_orders=True,
#         finalize_trades=True)
#     stats = bt.run()
#     profit = stats['Equity Final [$]']-100000
# #    print(stats)
# #    scores.append(profit)
# #    sharpe.append(stats['Sharpe Ratio'])
# #    sortino.append(stats['Sortino Ratio'])
# #    calmar.append(stats['Calmar Ratio'])

#     return profit, stats['Sharpe Ratio'], stats['Sortino Ratio'], stats['Calmar Ratio'], y_series


class Strategy2_opt_daytrading(Strategy):

    def init(self):
        # In init() and in next() it is important to call the
        # super method to properly initialize the parent classes
        super().init()

        # Set trailing stop-loss to 2x ATR using
        # the method provided by `TrailingStrategy`
#        self.set_trailing_sl(2)
    def next(self):
#        print(self.data.df['labeling_multi'].iloc[-1])
        if self.data.DaytradingExit[-1]:
            if self.position:      # daytrading
                self.position.close()
            return
        if self.data.y_pred[-1]>=0.5:
            # if self.position.is_short:
            #    self.position.close()
            #    return
            if not self.position:
#                self.buy(size=math.floor(100000/self.data.Close[-1]), limit=None, stop=None, sl=(1-self.data.sl[-1])*self.data.Close[-1], tp=(1+self.data.tp[-1])*self.data.Close[-1], tag=None)
                self.buy(size=1, limit=None, stop=None, sl=(1-self.data.sl[-1])*self.data.Close[-1], tp=(1+self.data.tp[-1])*self.data.Close[-1], tag=None)

        if self.data.y_pred[-1]<=-0.5:
            # if self.position.is_long:
            #    self.position.close()
            #    return
            if not self.position:
#                self.sell(size=math.floor(100000/self.data.Close[-1]), limit=None, stop=None, tp=(1-self.data.tp[-1])*self.data.Close[-1], sl=(1+self.data.sl[-1])*self.data.Close[-1], tag=None)
                self.sell(size=1, limit=None, stop=None, tp=(1-self.data.tp[-1])*self.data.Close[-1], sl=(1+self.data.sl[-1])*self.data.Close[-1], tag=None)



######################################################## EVALS STRATEGY ########################################################


def do_backtest_Strategy2_evals(X_train, X_test, y_train, y_test, data_target, params, weighting):
    rand_int = randint(1000000, 2000000)
    params_gpu = {
        'num_class': 3, 'device': 'gpu',
        'learning_rate': params['learning_rate'],
        # 'n_estimators': params['n_estimators'],
        'max_depth': params['max_depth'],
        'subsample': params['subsample'],
        'gamma': params['gamma'],
        'objective': 'multi:softprob',
        'seed': rand_int,
        'eval_metric': 'auc', #'merror',
        # 'verbose_eval': 1000
    }

    sample_weights = compute_sample_weight(
        class_weight='balanced',
        y=y_train
    )
    dtrain_gpu = DMatrix(X_train, label=y_train, weight=sample_weights)
    dtest_gpu = DMatrix(X_test, label=y_test)
    # print("dtrain_gpu: ", dtrain_gpu.shape)
    # print("dtest_gpu: ", dtest_gpu.shape)
    # print("sample_weights: ", sample_weights.shape)
    # print("y_train: ", y_train.shape)
    # print("y_test: ", y_test.shape)
    # print("X_train: ", X_train.shape)
    # print("X_test: ", X_test.shape)
    # print("data_target: ", data_target.shape)
    # print("weighting: ", weighting)

    print("Training model...")
    model_gpu = train(params_gpu, dtrain_gpu, num_boost_round=params['n_estimators'])
    # importance_scores = model_gpu.get_score(importance_type='weight')
    # sorted_by_values = dict(sorted(importance_scores.items(), key=lambda item: item[1]))
    # json.dump( sorted_by_values, open( f'importance_scores_optim_{rand_int}.json', 'w' ) )

    # if X_test.count() < 5:
    #     return pd.Series(np.zeros(X_test.count()), index=X_test.index, name="y_pred"), EMPTY_STATS, model_gpu

    print("Predicting...")
    y_pred = model_gpu.predict(dtest_gpu)
    # print("y_pred: ", y_pred)
    # print("Predicted.")
    # print("Creating y_series...")
    y_pred_expected = np.matmul(y_pred, np.array([[-1],[0],[1]]))
    # y_series = pd.Series(y_pred-1, index=X_test.index, name="y_pred")
    # y_series = pd.Series(y_pred_expected.flatten(), index=X_test.index, name="y_pred").rolling(window=params['pred_avg_period'], min_periods=1).mean()
    y_series = pd.Series(y_pred_expected.flatten(), index=X_test.index, name="y_pred").ewm(span=params['pred_ewm_span'], adjust=False).mean()


    print("Joining y_series to data_target...")
    data_target = data_target.join(y_series, how='left')

    print("Backtesting...")
    # data_target.to_csv(f'data_target_optim_{rand_int}.csv')

    bt = Backtest(data_target,
        Strategy2_opt_evals,
        cash=100000,
        spread=0,
        commission=0.00008,
        margin=1,
        trade_on_close=False,
        hedging=False,
        exclusive_orders=True,
        finalize_trades=True)
    stats = bt.run()

    return y_series, stats #, model_gpu



class Strategy2_opt_evals(Strategy):

    def init(self):
        # In init() and in next() it is important to call the
        # super method to properly initialize the parent classes
        super().init()

    def next(self):

        if self.position:
            if self.data.DaytradingExit[-1]:
                self.position.close()
                return
            for trade in self.trades:
                if trade.is_long:
                    trade.sl = max(trade.sl or -np.inf, self.data.High[-1] - self.data.sl[-1])
                elif trade.is_short:
                    trade.sl = min(trade.sl or np.inf, self.data.Low[-1] + self.data.sl[-1])
            return
        
        if self.data.y_pred[-1]>=0.5:
            # if self.position.is_short:
            #    self.position.close()
            #    return
            # if not self.position:
#                self.buy(size=math.floor(100000/self.data.Close[-1]), limit=None, stop=None, sl=(1-self.data.sl[-1])*self.data.Close[-1], tp=(1+self.data.tp[-1])*self.data.Close[-1], tag=None)
            self.buy(size=1, limit=None, stop=None, sl=self.data.Close[-1] - self.data.sl[-1], tp=self.data.Close[-1] + self.data.tp[-1], tag=None)
            return

        if self.data.y_pred[-1]<=-0.5:
            # if self.position.is_long:
            #    self.position.close()
            #    return
            # if not self.position:
#                self.sell(size=math.floor(100000/self.data.Close[-1]), limit=None, stop=None, tp=(1-self.data.tp[-1])*self.data.Close[-1], sl=(1+self.data.sl[-1])*self.data.Close[-1], tag=None)
            self.sell(size=1, limit=None, stop=None, tp=self.data.Close[-1] - self.data.tp[-1], sl=self.data.Close[-1] + self.data.sl[-1], tag=None)





def do_backtest_Strategy2_training(X_train, y_train, params):
    rand_int = randint(1000000, 2000000)
    params_gpu = {
        'num_class': 3, 'device': 'gpu',
        'learning_rate': params['learning_rate'],
        # 'n_estimators': params['n_estimators'],
        'max_depth': params['max_depth'],
        'subsample': params['subsample'],
        'gamma': params['gamma'],
        'objective': 'multi:softprob',
        'seed': rand_int,
        'eval_metric': 'auc', #'merror',
        # 'verbose_eval': 1000
    }

    sample_weights = compute_sample_weight(
        class_weight='balanced',
        y=y_train
    )
    dtrain_gpu = DMatrix(X_train, label=y_train, weight=sample_weights)
    # print("dtrain_gpu: ", dtrain_gpu.shape)
    # print("dtest_gpu: ", dtest_gpu.shape)
    # print("sample_weights: ", sample_weights.shape)
    # print("y_train: ", y_train.shape)
    # print("y_test: ", y_test.shape)
    # print("X_train: ", X_train.shape)
    # print("X_test: ", X_test.shape)
    # print("data_target: ", data_target.shape)
    # print("weighting: ", weighting)

    print("Training model...")
    model_gpu = train(params_gpu, dtrain_gpu, num_boost_round=params['n_estimators'])
    # importance_scores = model_gpu.get_score(importance_type='weight')
    # sorted_by_values = dict(sorted(importance_scores.items(), key=lambda item: item[1]))
    # json.dump( sorted_by_values, open( f'importance_scores_optim_{rand_int}.json', 'w' ) )

    # if X_test.count() < 5:
    #     return pd.Series(np.zeros(X_test.count()), index=X_test.index, name="y_pred"), EMPTY_STATS, model_gpu


    return model_gpu




# class TrailingStrategy(SignalStrategy):

#     __sl_amount = 100
#     def set_trailing_sl(self, sl_amount: float = 100):
#         """
#     Set the trailing stop loss as $n below the current price (for long positions)
#         Works for future bars only
#         """
#         self.__sl_amount = sl_amount


#     def init(self):
#         # In init() and in next() it is important to call the
#         # super method to properly initialize the parent classes
#         super().init()

#         # Set trailing stop-loss to 2x ATR using
#         # the method provided by `TrailingStrategy`
# #        self.set_trailing_sl(2)
#     def next(self):
# #        print(self.data.df['labeling_multi'].iloc[-1])

#         for trade in self.trades:
#             if trade.is_long:
#                 trade.sl = max(trade.sl or -np.inf, self.data.High[-1] - self.__sl_amount)
#             elif trade.is_short:
#                 trade.sl = min(trade.sl or np.inf, self.data.Low[-1] + self.__sl_amount)

# class Strategy2_opt_evals(TrailingStrategy):

#     def init(self, sl_amount: float = 100, tp_amount: float = 150):
#         # In init() and in next() it is important to call the
#         # super method to properly initialize the parent classes
#         super().init()
#         self.set_trailing_sl(sl_amount)

#     def next(self):
# #        print(self.data.df['labeling_multi'].iloc[-1])
#         super().next()

#         if self.position:
#             if self.data.DaytradingExit[-1]:
#                 self.position.close()
#                 return
#             return
        
#         if self.data.y_pred[-1]>=0.5:
#             # if self.position.is_short:
#             #    self.position.close()
#             #    return
#             # if not self.position:
# #                self.buy(size=math.floor(100000/self.data.Close[-1]), limit=None, stop=None, sl=(1-self.data.sl[-1])*self.data.Close[-1], tp=(1+self.data.tp[-1])*self.data.Close[-1], tag=None)
#             self.buy(size=1, limit=None, stop=None, sl=self.data.Close[-1]-self.__sl_amount, tp=self.data.Close[-1]+self.__tp_amount, tag=None)

#         if self.data.y_pred[-1]<=-0.5:
#             # if self.position.is_long:
#             #    self.position.close()
#             #    return
#             # if not self.position:
# #                self.sell(size=math.floor(100000/self.data.Close[-1]), limit=None, stop=None, tp=(1-self.data.tp[-1])*self.data.Close[-1], sl=(1+self.data.sl[-1])*self.data.Close[-1], tag=None)
#             self.sell(size=1, limit=None, stop=None, tp=self.data.Close[-1]-self.__tp_amount, sl=self.data.Close[-1]+self.__sl_amount, tag=None)
