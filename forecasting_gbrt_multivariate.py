from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.decomposition import PCA

from reservoir_computing.reservoir import Reservoir
from reservoir_computing.utils import make_forecasting_dataset
from reservoir_computing.datasets import PredLoader
from sklearn.preprocessing import StandardScaler

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
    "hour_sin", "hour_cos",

    "Close",

]


df = pd.read_csv("X.csv", index_col=0, parse_dates=True)
df_X = df.iloc[20000:][cols]
df_y = df.iloc[20000:]["Close"]

print(df_X.shape)
print(df_y.shape)


# Generate training and test datasets
# Xtr, Ytr, Xte, Yte, scaler = make_forecasting_dataset(df_X_d,
#                                                       horizon=24, # forecast horizon of 2h ahead
                                                    #   test_percent = 0.1)


dates = np.unique(df_X.index.date)
print(len(dates))
dates = [d for d in dates if d.weekday() in [0, 1, 2, 3, 4]] # only keep weekdays
dates = sorted(dates)
print(len(dates))

X_sum = []
Y_sum = []

for d in dates[:-1]:
    print(d)
    mask = (df_X.index.date == d) & (df_X.index.hour >= 6)
    mask_d1 = (df_X.index.date == d+timedelta(days=1)) & (df_X.index.hour >= 6)
    X = df_X[mask].iloc[:12*12].to_numpy()
    Y = df_y[mask_d1].iloc[:12*12].to_numpy()
    # Xte = df_X[mask].iloc[6*12:8*12].to_numpy()
    # Yte = df_y[mask].iloc[6*12+horizon:8*12+horizon].to_numpy()

    if X.shape[0] != Y.shape[0]:
        continue
    if len(Y) == 0:
        continue
    if len(X) == 0:
        continue

    print(X.shape, Y.shape)
    X_sum.append(X)
    Y_sum.append(Y)
    # print(Xte.shape, Yte.shape)
    # Xte_sum.append(Xte)
    # Yte_sum.append(Yte)


print("X_sum[0].shape: ", X_sum[0].shape)
print("Y_sum[0].shape: ", Y_sum[0].shape)
# print("Xte_sum[0].shape: ", Xte_sum[0].shape)
# print("Yte_sum[0].shape: ", Yte_sum[0].shape)

split_index = int(len(X_sum) * 0.9)

Xtr = np.concatenate(X_sum[:split_index], axis=0)
Ytr = np.concatenate(Y_sum[:split_index], axis=0)
Xte = np.concatenate(X_sum[split_index:], axis=0)
Yte = np.concatenate(Y_sum[split_index:], axis=0)

print("Xtr.shape: ", Xtr.shape)
print("Ytr.shape: ", Ytr.reshape(-1, 1).shape)
print("Xte.shape: ", Xte.shape)
print("Yte.shape: ", Yte.reshape(-1, 1).shape)



scalerX = StandardScaler()
scalerY = StandardScaler()


print("Scaled inputs:")
# Fit scaler on training set
Xtr = scalerX.fit_transform(Xtr)
print("Xtr.shape: ", Xtr.shape)
print("Xtr: ", Xtr)
# Transform the rest
Ytr = scalerY.fit_transform(Ytr.reshape(-1, 1))
print("Ytr.shape: ", Ytr.shape)
print("Ytr: ", Ytr)
Xte = scalerX.transform(Xte)
print("Xte.shape: ", Xte.shape)
print("Xte: ", Xte)


res = Reservoir(n_internal_units=1000, 
                spectral_radius=0.95, 
                leak=None, 
                connectivity=0.25, 
                input_scaling=0.1, 
                noise_level=0.0, 
                circle=False)   


print("Xtr [N,T,V].shape: ", Xtr[None,:,:].shape)
print("Xte [N,T,V].shape: ", Xte[None,:,:].shape)
print("Xtr [N,T,V]: ", Xtr[None,:,:])
print("Xte [N,T,V]: ", Xte[None,:,:])


n_drop=0
states_tr = res.get_states(Xtr[None,:,:], n_drop=n_drop, bidir=False)
states_te = res.get_states(Xte[None,:,:], n_drop=n_drop, bidir=False)


print("states_tr.shape: ", states_tr.shape)
print("states_te.shape: ", states_te.shape)


pca = PCA(n_components=100)
states_tr = pca.fit_transform(states_tr[0])
states_te = pca.transform(states_te[0])


print("states_tr_pca.shape: ", states_tr.shape)
print("states_te_pca.shape: ", states_te.shape)


# Fit the ridge regression model
ridge = Ridge(alpha=1.0) 
ridge.fit(states_tr, Ytr[n_drop:,:])

# Compute the predictions
Yhat = ridge.predict(states_te)

print("Yhat.shape: ", Yhat.shape)
print("Yhat: ", Yhat)


Yhat_transformed = scalerY.inverse_transform([Yhat]).T
print("Yhat_transformed.shape: ", Yhat_transformed.shape)
print("Yhat_transformed: ", Yhat_transformed)
print("Yte: ", Yte.shape)
print("Yte: ", Yte)

fig = plt.figure(figsize=(14,4))
plt.plot(Yte[n_drop:], 'k--', label="True", linewidth=2)
plt.plot(Yhat_transformed, label="Predicted")
plt.grid()
plt.legend()
plt.title("True vs predicted electricity load")
# plt.show()
plt.savefig("predictions_ridge2.png")


# GBRT

# Quantile 0.5
max_iter = 100
gbrt_median = HistGradientBoostingRegressor(
    loss="quantile", quantile=0.5, max_iter=max_iter)
gbrt_median.fit(states_tr, Ytr[n_drop:,0])
median_predictions = gbrt_median.predict(states_te)

# Quantile 0.05
gbrt_percentile_5 = HistGradientBoostingRegressor(
    loss="quantile", quantile=0.05, max_iter=max_iter)
gbrt_percentile_5.fit(states_tr, Ytr[n_drop:,0])
percentile_5_predictions = gbrt_percentile_5.predict(states_te)

# Quantile 0.95
gbrt_percentile_95 = HistGradientBoostingRegressor(
    loss="quantile", quantile=0.95, max_iter=max_iter)
gbrt_percentile_95.fit(states_tr, Ytr[n_drop:,0])
percentile_95_predictions = gbrt_percentile_95.predict(states_te)



# Plot the results
fig = plt.figure(figsize=(14,4))
plt.plot(Yte[n_drop:], 'k--', label="True", linewidth=2)
plt.plot(scalerY.inverse_transform(median_predictions[:,None]), label="Median prediction", color="tab:blue")
plt.fill_between(np.arange(len(Yte[n_drop:])), scalerY.inverse_transform(percentile_5_predictions[:,None]).ravel(), scalerY.inverse_transform(percentile_95_predictions[:,None]).ravel(), alpha=0.3, label="90% CI", color="tab:blue")
plt.grid()
plt.legend()
plt.title("Predicted electricity load using Gradient Boosting Regression Trees")
# plt.show()
plt.savefig("predictions_gbrt2.png")



