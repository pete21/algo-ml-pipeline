from datetime import date
import logging
import pprint
import dvc.api
import numpy as np
import pandas as pd

from numpy.lib.stride_tricks import sliding_window_view

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from reservoir_computing.modules import RC_model
from reservoir_computing.utils import compute_test_scores

from src.data_utils.dynamic_features import dynamic_features
from src.data_utils.features import build_target
from src.data_utils.utils import get_dates
from src.model.model_building import load_data
from src.trade.model_inference_trade import MODEL_PARAMS_URL, fetch_model_params
# from reservoir_computing.datasets import ClfLoader

WINDOW_SIZE = 24            # size of the window for the sliding window view, 24*5m = 2h

# np.random.seed(0) # For reproducibility

# cols = [
#     "sma_cross", "sma_cross_15m", "sma_cross_1h", "lowerband_r", "lowerband_r_15m", "lowerband_r_1h", "stochk_slope", "stochk_slope_15m", "stochk_slope_1h",
#     "ha_slope_2", "ha_slope_2_15m", "ha_slope_2_1h", "ema_cross", "ema_cross_15m", "ema_cross_1h",
#     "ha_slope_10", "ha_slope_10_15m", "ha_slope_10_1h",

#     "ppo", "ppo_15m", "ppo_1h", "upperband_r", "upperband_r_15m", "upperband_r_1h", "sar_r", "sar_r_15m", "sar_r_1h",
#     "kama_diff", "kama_diff_15m", "kama_diff_1h",
#     # "macd", "macd_15m", "macd_1h", "macd_signal", "macd_signal_15m", "macd_signal_1h",
#     # "Close", "Open", "High", "Low",

#     "di_plus", "di_plus_15m", "di_plus_1h", "di_minus", "di_minus_15m", "di_minus_1h", "cci_ha", "cci_ha_15m", "cci_ha_1h", "rsi", "rsi_15m", "rsi_1h", "rsi_ha", "rsi_ha_15m", "rsi_ha_1h", "di_diff", "di_diff_15m", "di_diff_1h", "willr", "willr_15m", "willr_1h",
#     "hour_sin", "hour_cos"

# ]


# Logging configuration
logger = logging.getLogger('rc')
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler('rc_log.log')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)



config = {}

# Hyperarameters of the reservoir
config['n_internal_units'] = 2000        # size of the reservoir
config['spectral_radius'] = 1        # largest eigenvalue of the reservoir
config['leak'] = 0.9                    # amount of leakage in the reservoir state update (None or 1.0 --> no leakage)
config['connectivity'] = 0.2           # percentage of nonzero connections in the reservoir
config['input_scaling'] = 0.1           # scaling of the input weights
config['noise_level'] = 0.01            # noise in the reservoir state update
config['n_drop'] = 5                    # transient states to be dropped
config['bidir'] = True                  # if True, use bidirectional reservoir
config['circle'] = True                # use reservoir with circle topology

# Dimensionality reduction hyperparameters
config['dimred_method'] = 'tenpca'      # options: {None (no dimensionality reduction), 'pca', 'tenpca'}
config['n_dim'] = 100                    # number of resulting dimensions after the dimensionality reduction procedure

# Type of MTS representation
config['mts_rep'] = 'reservoir'         # MTS representation:  {'last', 'mean', 'output', 'reservoir'}
config['w_ridge_embedding'] = 10.0      # regularization parameter of the ridge regression

# Type of readout
# config['readout_type'] = 'lin'          # readout used for classification: {'lin', 'mlp', 'svm'}
# config['w_ridge'] = 5.0                 # regularization of the ridge regression readout

# Type of readout
# config['readout_type'] = 'svm'          # readout used for classification
# config['svm_gamma'] = 5e-3              # bandwith of the RBF kernel
# config['svm_C'] = 10.0                  # regularization for SVM hyperplane

# Type of readout
config['readout_type'] = 'mlp'          # readout used for classification
config['mlp_layout'] = (128,64,32)          # neurons in each MLP layer
config['num_epochs'] = 2000             # number of epochs 
config['w_l2'] = 1e-3                  # weight of the L2 regularization
config['nonlinearity'] = 'identity'         # type of activation function {'relu', 'tanh', 'logistic', 'identity'}

pprint.pprint(config)



# Xtr, Ytr, Xte, Yte = ClfLoader().get_data('Japanese_Vowels')
# print(Xtr.shape, Ytr.shape, Xte.shape, Yte.shape)
# print(Xtr)
# print(Ytr)



# df_X = pd.read_csv("X.csv", index_col=0, parse_dates=True)
# df_X = df_X.iloc[0:][cols]

# df_y = pd.read_csv("y.csv", index_col=0, parse_dates=True)
# df_y = df_y.iloc[0:]["labeling_multi"]
# # df_y = df_y["labeling_multi"]


params = dvc.api.params_show('params.yaml')['model_building']
print(f"Params: {params}")
# model_params = load_json_params(os.path.join(root_dir, 'model_params.json'), logger=logger)


# Load the preprocessed data from the interim directory
data = load_data(data_path=params['data_path'], params=params)

unique_dates, unique_weekdates = get_dates(data, params['index_base'])
# cutoff_date = data[params['index_base']].index.date.min()+pd.Timedelta(21, "D")
cutoff_date = date(2022, 1, 1)


model_params = fetch_model_params(url=MODEL_PARAMS_URL, logger=logger)
print(f"Loaded model params: {model_params}")
model_params['hour_range_start'] = 4*60
model_params['hour_range_stop'] = 18*60


cutoff_date_2 = cutoff_date - pd.Timedelta(14, "D")

for i in params['indexes_higher'] + [params['index_base']]:
    data[i] = data[i].loc[data[i].index.date>=cutoff_date_2]
    # print(data[i])

ml_data = {}
ml_data[params['index_base']] = dynamic_features(data[params['index_base']], model_params, params['timeframe_scalers'][params['index_base']], col_close="Close", col_high="High", col_low="Low")

ml_data[params['index_base']].loc[:,"labeling_binary"], ml_data[params['index_base']].loc[:,"labeling_dual_ema"], ml_data[params['index_base']].loc[:,"labeling_multi"] = build_target(ml_data[params['index_base']], \
    open_col='Open', high_col='High', low_col='Low', high_time_col="high_time", low_time_col="low_time", \
    tp=model_params['target_tp'], ema_period=model_params['ema_period'], ema_reversed_period=model_params['ema_reversed_period'], \
    threshold_long=model_params['threshold_long']-0.1, threshold_short=model_params['threshold_short']+0.1)

ml_data[params['index_base']] = ml_data[params['index_base']].loc[(ml_data[params['index_base']]['minute_of_day']>=model_params['hour_range_start']) & (ml_data[params['index_base']]['minute_of_day']<model_params['hour_range_stop'])]
ml_data[params['index_base']] = ml_data[params['index_base']][params['list_X'] + [params['col_y'][0]] + ["Open", "High", "Low", "date_merge", 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'minute_of_day']]
# print(ml_data[params['index_base']])
print(ml_data[params['index_base']].columns)
# target = ml_data[index_b].loc[(ml_data[index_b].index.hour>=parameters['hour_range_start']) & (ml_data[index_b].index.hour<=parameters['hour_range_start']+10), [y_col]]


for i in params['indexes_higher']:
    ml_data[i] = dynamic_features(data[i], model_params, params['timeframe_scalers'][i], col_close="Close", col_high="High", col_low="Low")
    # ml_data[i]['local_date'] = ml_data[i].index.tz_localize('UTC').tz_convert(local_timezone)
    # ml_data[i]['minute_of_day'] = ml_data[i]['local_date'].dt.hour * 60 + ml_data[i]['local_date'].dt.minute
    # ml_data[i] = ml_data[i].loc[(ml_data[i]['minute_of_day']>=model_params['hour_range_start']) & (ml_data[i]['minute_of_day']<model_params['hour_range_stop'])]
    ml_data[i] = ml_data[i][params['list_X'] + ["Open", "High", "Low", "date_merge"]]
    # print(ml_data[i])
    # print(ml_data[i].columns)

for i in params['indexes_higher'] + [params['index_base']]:
    ml_data[i] = ml_data[i].loc[ml_data[i].index.date>=cutoff_date]
    # print(ml_data[i])


dates = np.unique(ml_data[params['index_base']].index.date)
print(len(dates))
dates = [d for d in dates if d.weekday() in [0, 1, 2, 3, 4]] # only keep weekdays
dates = sorted(dates)
print(len(dates))


# Transform columns

#  'sine','sine_diff_slope_2', 'sine_diff_slope_5', 'kama_trend_slow_diff', 'kama_trend_fast_diff',
#       'kama_trend_slow_diff2', 'kama_trend_fast_diff2','sma_cross', 'lowerband_r',
# 'ha_slope_2', 'ema_cross', 'log_ret_ha_1', 'log_ret_ha_2',
# 'log_ret_ha_3', 'ppo', 'upperband_r', 'sar_r',
#  'stochd', 'kama_diff',
    #    'bearish_gap', 'rsi_slope', 'sine_diff', 'stochrsid', 'stochrsik',
    #    'bullish_gap', 'ha_slope_10', 'r_kijun_sen', 'r_tenkan_sen',
    #    'displacement_hull',
    # 'dc_market_regime_ema_log',
# 'stochk', 'ema_ha_sign',
# 'ema_ha_wickstrength', 'gap_hull',
# 'di_plus',
# 'di_minus', 'cci_ha', 'rsi', 'rsi_ha', 'di_diff', 'willr',
# 'long_pivot', 'short_pivot',
# 'permutation_entropy', 'skew', 'petrosian_fd',
# 'date_merge'

# ct = {}
# for i in params['indexes_higher'] + [params['index_base']]:
#     ct[i] = ColumnTransformer([("minmaxscaler", MinMaxScaler(), ['ha_close', 'Close', 'tenkan_sen', 'kijun_sen', 'stochk_slope', 
#         'kama_trend_slow', 'kama_trend_fast', 'macdhist',
#         'displacement_hull_slope',  'gap_hull_slope', 
#         'close_wavelet_rolling', 
#         'atr', 'adx', 'Open', 'High', 'Low',
#         ])], remainder='passthrough')

Xy_sum = []
# ct_sum = []

transformed_columns = ['ha_close', 'Close', 'tenkan_sen', 'kijun_sen', 'stochk_slope', 
        'kama_trend_slow', 'kama_trend_fast', 'macdhist',
        'displacement_hull_slope',  'gap_hull_slope', 
        'close_wavelet_rolling', 
        'atr', 'adx', 'Open', 'High', 'Low',
        ]

ct = ColumnTransformer([("minmaxscaler", MinMaxScaler(), transformed_columns)], remainder='passthrough')

X={}
# y_sum=[]

min_window_end_minute = 540 #10*60
num_windows_per_day = 4

window_step_minute = 15

max_window_end_minute = min_window_end_minute+num_windows_per_day*window_step_minute

print(len(dates))

for d in dates:
    # print(d)

    X_columntransform = ml_data[params['index_base']].loc[(ml_data[params['index_base']].index.date == d) & (ml_data[params['index_base']]['minute_of_day']<min_window_end_minute+WINDOW_SIZE*params['timeframe_minutes'][params['index_base']])]
    
    if X_columntransform.index.size < 420/params['timeframe_minutes'][params['index_base']]:
        print("X_columntransform window size too small:", d,  X_columntransform.index.size)
        continue
    ct.fit(X_columntransform.drop(columns=['date_merge','minute_of_day','labeling_multi']))             # should refit the scaler for each day
    # ct_sum.append(ct)
    
    # print("ct input columns: ", X_columntransform.columns)
    # print("ct output columns: ", ct.get_feature_names_out())
    
    for window_end in range(min_window_end_minute, max_window_end_minute, window_step_minute):

        X[params['index_base']] = ml_data[params['index_base']].loc[(ml_data[params['index_base']].index.date == d) & (ml_data[params['index_base']]['minute_of_day']<window_end) & (ml_data[params['index_base']]['minute_of_day']>=window_end-30-WINDOW_SIZE*params['timeframe_minutes'][params['index_base']])].iloc[-WINDOW_SIZE:]
        # print(X[params['index_base']].index.size)
        if X[params['index_base']].index.size < WINDOW_SIZE:
            print("X window size too small:", d, X[params['index_base']].index.size)
            continue
        # print(X[params['index_base']])
        date_merge = X[params['index_base']]['date_merge'].iloc[-1]

        y = X[params['index_base']]['labeling_multi'].iloc[-1:].to_numpy()
        # y_sum.append(y+1)
        # continue

        X[params['index_base']] = X[params['index_base']].drop(columns=['date_merge','minute_of_day','labeling_multi'], errors='ignore')
        X[params['index_base']] = ct.transform(X[params['index_base']])

        # print(X[params['index_base']].shape)

        for i in params['indexes_higher']:
            X[i] = ml_data[i].loc[(ml_data[i]['date_merge']<=date_merge)].iloc[-WINDOW_SIZE:]
            # print(X[i])
            X[i] = X[i].drop(columns=['date_merge','minute_of_day'], errors='ignore')
            X[i]['hour_sin']=0
            X[i]['hour_cos']=0
            X[i]['dow_sin']=0
            X[i]['dow_cos']=0
            X[i] = ct.transform(X[i])

            num_columns = X[i].shape[1]
            X[i] = np.delete(X[i], np.s_[num_columns-4:], axis=1)
            # print(X[i].shape)
        if any(X[i].shape[0] < WINDOW_SIZE for i in params['indexes_higher']):
            X={}
            continue

        Xy_dict = {}
        for i in params['indexes_higher'] + [params['index_base']]:
            Xy_dict[i] = X[i]
        Xy_dict['date_merge'] = date_merge
        Xy_dict['y'] = y+1 # +1 because the labels are 0, 1, 2

        Xy_sum.append(Xy_dict)


# y_flat = np.concatenate(y_sum).reshape(-1)
# print(y_flat)
# hist, bins = np.histogram(y_flat)
# print("y distribution: ", hist, bins)
# exit()

for i in [params['index_base']] + params['indexes_higher']:
    print(Xy_sum[0][i].shape)
    print(Xy_sum[0][i])

print("Number of windows:", len(Xy_sum))

split_index = int(len(Xy_sum) * 0.85)
print("Split index:", split_index)


Xy_train = Xy_sum[:split_index]
Xy_test = Xy_sum[split_index:]

print(len(Xy_train))
print(len(Xy_test))


# classifier = {}
# for i in params['indexes_higher'] + [params['index_base']]:
#     classifier[i] = RC_model(**config)
#     print(f"Training the model for {i}...")
#     tr_time = classifier[i].fit(X_train[i], y_train[i]) 
#     print(f"Computing predictions on test data for {i}...")
#     pred_class = classifier[i].predict(X_test[i])
#     print(pred_class)
#     accuracy, f1 = compute_test_scores(pred_class, y_test[i])
#     print(f"Accuracy = {accuracy:.3f}, F1 = {f1:.3f}")



Xtr_concat = []
ytr_concat = []
for i in range(len(Xy_train)):
    Xtr_concat.append(np.concatenate((Xy_train[i][5],Xy_train[i][7],Xy_train[i][10]), axis=1))
    ytr_concat.append(Xy_train[i]['y'])

Xte_concat = []
yte_concat = []
for i in range(len(Xy_test)):
    Xte_concat.append(np.concatenate((Xy_test[i][5],Xy_test[i][7],Xy_test[i][10]), axis=1))
    yte_concat.append(Xy_test[i]['y'])



Xtr_concat = np.asarray(Xtr_concat)
print(Xtr_concat.shape)
# print(Xtr_concat)

ytr_concat = np.asarray(ytr_concat)
print(ytr_concat.shape)
# print(ytr_concat)

Xte_concat = np.asarray(Xte_concat)
print(Xte_concat.shape)
# print(Xte_concat)

yte_concat = np.asarray(yte_concat)
print(yte_concat.shape)
# print(yte_concat)


yte_concat_flat = yte_concat.reshape(-1)
print(yte_concat_flat)

hist, bins = np.histogram(yte_concat_flat)
print("y_test distribution: ", hist, bins)

# One-hot encoding for labels
onehot_encoder = OneHotEncoder(sparse_output=False)
# print(ytr_concat)
ytr_concat = onehot_encoder.fit_transform(ytr_concat)
# print(ytr_concat)
yte_concat = onehot_encoder.transform(yte_concat)
# print(yte_concat)

# print(y_train)

classifier =  RC_model(**config)

print("Training the model...")
# Train the model
tr_time = classifier.fit(Xtr_concat, ytr_concat)

print("Computing predictions on test data...")
# Compute predictions on test data
pred_class = classifier.predict(Xte_concat)

print(pred_class)
hist, bins = np.histogram(pred_class)
print("pred_class distribution: ", hist, bins)

accuracy, f1 = compute_test_scores(pred_class, yte_concat)
print(f"Accuracy = {accuracy:.3f}, F1 = {f1:.3f}")

print("Mean absolute difference between predicted and actual labels:")
diff = np.abs(pred_class - yte_concat_flat).mean()
squared_diff = np.square(pred_class - yte_concat_flat).mean()
print(f"Mean absolute difference = {diff:.3f}")
print(f"Mean squared difference = {squared_diff:.3f}")

