import numpy as np
from backtesting import Backtest
from backtesting.lib import Strategy

# from xgboost import XGBClassifier
from scipy.special import softmax
from xgboost import DMatrix, train
from sklearn.utils.class_weight import compute_sample_weight
from random import randint
import pandas as pd
import matplotlib.pyplot as plt

MULTI_CLASS_VALUES = np.array([[-0.9],[0],[0.9]])


def custom_distance_softmax(preds: np.ndarray, dtrain: DMatrix) -> tuple[np.ndarray, np.ndarray]:
    labels = dtrain.get_label().astype(int)
    # num_class = 3
    
    # Reshape predictions to (N, num_class)
    # preds = preds.reshape(-1, num_class)
    
    # Apply softmax to convert raw margins to probabilities
    exp_preds = np.exp(preds - np.max(preds, axis=1, keepdims=True))
    prob = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
    
    # Define penalty matrix [True Label][Predicted Label]
    # Extreme mistakes (0 vs 2) get a heavy multiplier (e.g., 5.0)
    penalty_matrix = np.array([
        [1.0, 1.01, 1.04],  # True 0: normal for 0/1, heavy for 2
        [1.02, 1.0, 1.02],  # True 1: normal penalties
        [1.04, 1.01, 1.0]   # True 2: heavy for 0, normal for 1/2
    ])
    
    # Extract specific weights for each sample's true label
    sample_penalties = penalty_matrix[labels] # Shape (N, num_class)
    
    # Standard multi-class gradient: prob - indicator
    # Multiply by the sample-specific penalties
    grad = prob.copy()
    grad[range(len(labels)), labels] -= 1.0
    grad = grad * sample_penalties
    
    # Standard Hessian approximation for multi-class
    hess = 2.0 * prob * (1.0 - prob) * sample_penalties
    
    return grad.flatten(), hess.flatten()


def tracking_metric(preds, dtrain):
    labels = dtrain.get_label().astype(int)
    # num_classes = 3
    
    # Reshape XGBoost outputs to (num_samples, num_classes)
    # preds = preds.reshape(-1, num_classes)
    
    # Convert raw outputs to probabilities and get the predicted classes
    probs = softmax(preds, axis=1)
    predictions = np.argmax(probs, axis=1)
    
    # 1. Count "Severe Flips" (True 0 predicted as 2, OR True 2 predicted as 0)
    severe_flips = np.sum(((labels == 0) & (predictions == 2)) | 
                          ((labels == 2) & (predictions == 0)))
    
    # 2. Count "False Breakouts" (True 1 predicted as 0, OR True 1 predicted as 2)
    false_breakouts = np.sum((labels == 1) & ((predictions == 0) | (predictions == 2)))
    
    # Note: XGBoost custom metrics return a (metric_name, metric_value) tuple.
    # To track multiple things, we can combine them into a single score or printout.
    # Here we return total critical errors, but we can print details to the console.
    
    total_critical_errors = 2*severe_flips + false_breakouts
    
    # Optional: Print live breakdown during training iterations
    # print(f" -> [Severe Flips: {severe_flips} | False Breakouts: {false_breakouts}]")
    
    return 'critical_errors', float(total_critical_errors)


def viterbi_sequence_decoder(probabilities: np.ndarray, extreme_jump_penalty: float = 0.2) -> np.ndarray:                   # Potentially introduces hindsight bias
    """
    Finds the optimal sequence of states (0, 1, 2) by balancing XGBoost 
    probabilities with a heavy penalty for sudden extreme jumps (0<->2).
    
    probabilities: np.ndarray of shape (N, 3)
    extreme_jump_penalty: Higher values make 0<->2 transitions harder to happen.
    """
    N = len(probabilities)
    num_states = 3
    
    # 1. Convert probabilities to negative log-likelihoods (cost to minimize)
    # Add a small epsilon to avoid log(0)
    eps = 1e-15
    emission_costs = -np.log(np.clip(probabilities, eps, 1.0 - eps))
    
    # 2. Define the transition cost matrix [from_state][to_state]
    # Standard steps (0->1, 1->2) have low cost. Extreme jumps (0<->2) have massive cost.
    transition_cost = np.array([
        [0, 0.1, extreme_jump_penalty],  # From 0 to: [0, 1, 2]
        [0.2, 0, 0.2],                  # From 1 to: [0, 1, 2]
        [extreme_jump_penalty, 0.1, 0]   # From 2 to: [0, 1, 2]
    ])
    
    # Dynamic Programming tables
    dp = np.zeros((N, num_states))
    path = np.zeros((N, num_states), dtype=int)
    
    # Initialize base case (first row)
    dp[0, :] = emission_costs[0, :]
    
    # Forward pass: compute minimum cost sequence up to time t
    for t in range(1, N):
        for s in range(num_states):
            # Combined cost: past cumulative cost + transition cost + current row cost
            total_costs = dp[t-1, :] + transition_cost[:, s] + emission_costs[t, s]
            dp[t, s] = np.min(total_costs)
            path[t, s] = np.argmin(total_costs)
            
    # Backward pass: trace the optimal path back to the beginning
    smoothed_labels = np.zeros(N, dtype=int)
    smoothed_labels[-1] = np.argmin(dp[-1, :])
    
    for t in range(N-2, -1, -1):
        smoothed_labels[t] = path[t+1, smoothed_labels[t+1]]
        
    return smoothed_labels


def viterbi_price_decoder(emission_probs, penalty=1.0):
    """
    Finds the optimal sequence of price changes avoiding extreme transitions.
    
    emission_probs: np.array of shape (N, 3) containing model probabilities
    penalty: float, the penalty applied to 0->2 and 2->0 transitions
    """
    N = emission_probs.shape[0]
    num_states = 3
    
    # 1. Convert probabilities to log-probabilities to avoid underflow
    # Add a tiny epsilon to prevent log(0)
    log_emissions = np.log(emission_probs + 1e-12)
    
    # 2. Define Log-Transition Matrix
    # 0 cost for normal transitions, negative penalty for extreme transitions
    # log_transitions = np.zeros((num_states, num_states))
    # log_transitions[0, 2] = -penalty  # UP -> DOWN
    # log_transitions[2, 0] = -penalty  # DOWN -> UP

    log_transitions = np.array([
        [-0.0, -0.1, -penalty],  # From 0 to: [0, 1, 2]
        [-0.15, -0.0, -0.15],                  # From 1 to: [0, 1, 2]
        [-penalty, -0.1, -0.0]   # From 2 to: [0, 1, 2]
    ])
    
    # 3. Initialize Viterbi matrix and backpointer matrix
    viterbi = np.zeros((N, num_states))
    backpointers = np.zeros((N, num_states), dtype=int)
    
    # Uniform prior for the first row (or use actual prior if known)
    viterbi[0] = log_emissions[0] - np.log(num_states)
    
    # 4. Forward Pass
    for t in range(1, N):
        for j in range(num_states):
            # Calculate score for all possible previous states leading to state j
            scores = viterbi[t-1] + log_transitions[:, j] + log_emissions[t, j]
            
            viterbi[t, j] = np.max(scores)
            backpointers[t, j] = np.argmax(scores)
            
    # 5. Backtracking Pass
    best_path = np.zeros(N, dtype=int)
    best_path[-1] = np.argmax(viterbi[-1])
    
    for t in range(N-2, -1, -1):
        best_path[t] = backpointers[t+1, best_path[t+1]]
        
    return best_path




# # Set strict thresholds for execution
# BUY_THRESHOLD = 0.5   # Must be 45% confident it goes Up
# SELL_THRESHOLD = 0.5  # Must be 45% confident it goes Down
# FLAT_MAX = 0.25        # Cannot be more than 25% confident it stays Flat

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
        'objective': 'multi:softprob', # 'reg:squarederror',
        'seed': rand_int,
        'eval_metric': ['mlogloss', 'merror'],
        'tree_method': 'hist'
        # 'verbose_eval': 1000
    }

    sample_weights = compute_sample_weight(
        class_weight='balanced',
        y=y_train
    )
    dtrain_gpu = DMatrix(X_train, label=y_train, weight=sample_weights)
    dtest_gpu = DMatrix(X_test)

    # Define evaluation list to monitor both sets
    evals_result = {}
    # watchlist = [(dtrain_gpu, 'train'), (dtest_gpu, 'val')]
    watchlist = [(dtrain_gpu, 'train')]


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
    model_gpu = train(params_gpu, dtrain_gpu, num_boost_round=params['n_estimators'],
        # obj=custom_distance_softmax, #distance_weighted_loss,
        # custom_metric=tracking_metric,  # Handles tracking
        # evals=watchlist,
        # evals_result=evals_result
        )
    # importance_scores = model_gpu.get_score(importance_type='weight')
    # sorted_by_values = dict(sorted(importance_scores.items(), key=lambda item: item[1]))
    # json.dump( sorted_by_values, open( f'importance_scores_optim_{rand_int}.json', 'w' ) )


    # Plotting the evaluation metrics
    # 1. Extract the critical error history for both train and validation sets
    # epochs = len(evals_result['train']['critical_errors'])
    # x_axis = range(0, epochs)

    # # 2. Set up the plot size and style
    # plt.figure(figsize=(10, 6))

    # # 3. Plot evaluation metrics lines
    # plt.plot(x_axis, evals_result['train']['critical_errors'], label='Training Errors', color='#1f77b4', linewidth=2)
    # # plt.plot(x_axis, evals_result['val']['critical_errors'], label='Validation Errors', color='#ff7f0e', linewidth=2)

    # # 4. Add clear labels and title
    # plt.title('Custom Critical Errors (Severe Flips + False Breakouts) over Epochs', fontsize=14, fontweight='bold')
    # plt.xlabel('Training Epochs (Boosting Rounds)', fontsize=12)
    # plt.ylabel('Total Count of Critical Mistakes', fontsize=12)

    # # 5. Add UI enhancements (Grid and Legend)
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.legend(fontsize=12, loc='upper right')

    # # 6. Show the final plot
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig(f'critical_errors_plot_{rand_int}.png')
    

    print("Predicting...")
    # 1. Get raw margin predictions for your test set
    y_pred = model_gpu.predict(dtest_gpu, output_margin=False)


    y_pred = np.matmul(y_pred, MULTI_CLASS_VALUES)

    # print("Viterbi sequence decoder...")              # Potentially introduces hindsight bias because it uses the full history of predictions to make the create optimal sequence
    # y_pred = viterbi_price_decoder(y_pred, penalty=0.2)-1
    # final_smoothed_predictions is now a 1D array of 0s, 1s, and 2s 
    # where 0->2 and 2->0 jumps have been actively suppressed.
    # print("Distribution of predictions: ", np.bincount(y_pred+1))

    # print("Final smoothed predictions: ", y_pred)
    hist, bins = np.histogram(y_pred)
    print("y_pred distribution: ", hist, bins)


    # # Extract individual column probabilities
    # prob_down = y_pred[:, 0]
    # prob_flat = y_pred[:, 1]
    # prob_up = y_pred[:, 2]

    # # Initialize all predictions as 'Flat' (1)
    # y_pred_expected = np.zeros(len(y_pred))

    # # Apply the logic
    # y_pred_expected[(prob_up > BUY_THRESHOLD) & (prob_flat < FLAT_MAX)] = 1   # Up
    # y_pred_expected[(prob_down > SELL_THRESHOLD) & (prob_flat < FLAT_MAX)] = -1 # Down


    # print("y_pred_expected: ", y_pred_expected)
    # y_series = pd.Series(y_pred-1, index=X_test.index, name="y_pred")
    # y_series = pd.Series(y_pred_expected.flatten(), index=X_test.index, name="y_pred").rolling(window=params['pred_avg_period'], min_periods=1).mean()
    y_series = pd.Series(y_pred.flatten(), index=X_test.index, name="y_pred").ewm(span=params['pred_ewm_span'], adjust=False).mean()

    # print("Joining y_series to data_target...")
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
        exclusive_orders=False,
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

        if not self.position:
            if self.data.y_pred[-1]>=0.5:
                self.buy(size=1, limit=None, stop=None, sl=(1-self.data.sl[-1])*self.data.Close[-1], tp=(1+self.data.tp[-1])*self.data.Close[-1], tag=None)
                return
            if self.data.y_pred[-1]<=-0.5:
                self.sell(size=1, limit=None, stop=None, tp=(1-self.data.tp[-1])*self.data.Close[-1], sl=(1+self.data.sl[-1])*self.data.Close[-1], tag=None)
                return

        # else:   # if there is position

        #     if self.position.pl_pct >= 0.002: # if profit percentage is greater than 0.15%, adjust stop-loss to sl/2 from current price (~ break even price)
        #         for trade in self.trades:
        #             if trade.is_long:
        #                 trade.sl = max(trade.sl or -np.inf, (1-self.data.sl[-1]/2)*self.data.Close[-1])
        #             elif trade.is_short:
        #                 trade.sl = min(trade.sl or np.inf, (1+self.data.sl[-1]/2)*self.data.Close[-1])

            # if self.position.pl_pct >= 0.0015: # if profit percentage is greater than 0.2%, open addon trade with half SL and half TP
            #     if self.data.y_pred[-1]>0.5 and self.position.size == 1:
            #         self.buy(size=1, limit=None, stop=None, sl=(1-self.data.sl[-1])*self.data.Close[-1], tp=(1+self.data.tp[-1]/2)*self.data.Close[-1], tag=None)
            #     elif self.data.y_pred[-1]<-0.5 and self.position.size == -1:
            #         self.sell(size=1, limit=None, stop=None, tp=(1-self.data.tp[-1]/2)*self.data.Close[-1], sl=(1+self.data.sl[-1])*self.data.Close[-1], tag=None)


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
        'eval_metric': ['mlogloss', 'merror'],
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
    y_pred_expected = np.matmul(y_pred, MULTI_CLASS_VALUES)
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
        'eval_metric': ['mlogloss', 'merror'],
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
