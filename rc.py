import pprint
import numpy as np
import pandas as pd

from numpy.lib.stride_tricks import sliding_window_view

from sklearn.preprocessing import OneHotEncoder
from reservoir_computing.modules import RC_model
from reservoir_computing.utils import compute_test_scores
# from reservoir_computing.datasets import ClfLoader

WINDOW_SIZE = 48            # size of the window for the sliding window view, 48*5m = 4h

# np.random.seed(0) # For reproducibility

cols = [
    "sma_cross", "sma_cross_15m", "sma_cross_1h", "lowerband_r", "lowerband_r_15m", "lowerband_r_1h", "stochk_slope", "stochk_slope_15m", "stochk_slope_1h",
    "ha_slope_2", "ha_slope_2_15m", "ha_slope_2_1h", "ema_cross", "ema_cross_15m", "ema_cross_1h",
    "ha_slope_10", "ha_slope_10_15m", "ha_slope_10_1h",

    "ppo", "ppo_15m", "ppo_1h", "upperband_r", "upperband_r_15m", "upperband_r_1h", "sar_r", "sar_r_15m", "sar_r_1h",
    "kama_diff", "kama_diff_15m", "kama_diff_1h",
    # "macd", "macd_15m", "macd_1h", "macd_signal", "macd_signal_15m", "macd_signal_1h",
    # "Close", "Open", "High", "Low",

    "di_plus", "di_plus_15m", "di_plus_1h", "di_minus", "di_minus_15m", "di_minus_1h", "cci_ha", "cci_ha_15m", "cci_ha_1h", "rsi", "rsi_15m", "rsi_1h", "rsi_ha", "rsi_ha_15m", "rsi_ha_1h", "di_diff", "di_diff_15m", "di_diff_1h", "willr", "willr_15m", "willr_1h",
    "hour_sin", "hour_cos"

]



config = {}

# Hyperarameters of the reservoir
config['n_internal_units'] = 500        # size of the reservoir
config['spectral_radius'] = 1        # largest eigenvalue of the reservoir
config['leak'] = 0.8                    # amount of leakage in the reservoir state update (None or 1.0 --> no leakage)
config['connectivity'] = 0.2           # percentage of nonzero connections in the reservoir
config['input_scaling'] = 0.1           # scaling of the input weights
config['noise_level'] = 0.01            # noise in the reservoir state update
config['n_drop'] = 5                    # transient states to be dropped
config['bidir'] = True                  # if True, use bidirectional reservoir
config['circle'] = True                # use reservoir with circle topology

# Dimensionality reduction hyperparameters
config['dimred_method'] = 'tenpca'      # options: {None (no dimensionality reduction), 'pca', 'tenpca'}
config['n_dim'] = 50                    # number of resulting dimensions after the dimensionality reduction procedure

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



df_X = pd.read_csv("X.csv", index_col=0, parse_dates=True)
df_X = df_X.iloc[0:][cols]

df_y = pd.read_csv("y.csv", index_col=0, parse_dates=True)
df_y = df_y.iloc[0:]["labeling_multi"]
# df_y = df_y["labeling_multi"]


print(df_X.shape, df_y.shape)
print(df_X)
print(df_y)



dates = np.unique(df_X.index.date)
print(len(dates))
dates = [d for d in dates if d.weekday() in [0, 1, 2, 3, 4]] # only keep weekdays
dates = sorted(dates)
print(len(dates))

X_sum = []
y_sum = []

for d in dates:
    print(d)
    mask = (df_X.index.date == d) & (df_X.index.hour >= 6) & (df_X.index.hour < 12)
    df_X_d = df_X[mask].to_numpy()
    df_y_d = df_y[mask].to_numpy()
    print(df_X_d.shape, df_y_d.shape)
    if len(df_X_d) < WINDOW_SIZE:
        continue
    # print(df_X_d)
    # print(df_y_d)    

    X = sliding_window_view(df_X_d, window_shape=(WINDOW_SIZE,), axis=0)
    X = np.swapaxes(X, 1, 2)
    # print(X.shape)
    # print(X)

    y = df_y_d[-X.shape[0]:].reshape(-1, 1)
    # print(y.shape)
    # print(y)

    X_sum.append(X)
    y_sum.append(y)

print("Number of days:")
print(len(X_sum))
assert len(X_sum) == len(y_sum)

split_index = int(len(X_sum) * 0.9)
print("Split index:")
print(split_index)
X_train = np.concatenate(X_sum[:split_index], axis=0)
y_train = np.concatenate(y_sum[:split_index], axis=0)
X_test = np.concatenate(X_sum[split_index:], axis=0)
y_test = np.concatenate(y_sum[split_index:], axis=0)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# print(X_train)
# print(y_train)
# print(X_test)
# print(y_test)

y_test_flat = y_test.reshape(-1)
print(y_test_flat)

# One-hot encoding for labels
onehot_encoder = OneHotEncoder(sparse_output=False)
y_train = onehot_encoder.fit_transform(y_train)
y_test = onehot_encoder.transform(y_test)

# print(y_train)

classifier =  RC_model(**config)

print("Training the model...")
# Train the model
tr_time = classifier.fit(X_train, y_train)

print("Computing predictions on test data...")
# Compute predictions on test data
pred_class = classifier.predict(X_test)
print(pred_class)
accuracy, f1 = compute_test_scores(pred_class, y_test)
print(f"Accuracy = {accuracy:.3f}, F1 = {f1:.3f}")

print("Mean absolute difference between predicted and actual labels:")
diff = np.abs(pred_class - y_test_flat).mean()
print(f"Mean absolute difference = {diff:.3f}")
