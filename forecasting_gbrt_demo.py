import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.decomposition import PCA

from reservoir_computing.reservoir import Reservoir
from reservoir_computing.utils import make_forecasting_dataset
from reservoir_computing.datasets import PredLoader

np.random.seed(0) # For reproducibility



downloader = PredLoader()
downloader.available_datasets(details=True)  # Describe available datasets


# Download data
ts_full = downloader.get_data("ElecRome")

# Resample the time series to hourly frequency
ts_hourly = np.mean(ts_full.reshape(-1, 6), axis=1)[:, None]
print("Resampled: ", ts_hourly.shape)

# Use only the first 3000 time steps
ts_small = ts_hourly[0:3000,:]
print("Resampled small: ", ts_small.shape)


print("ts_small.shape: ", ts_small.shape)
print("ts_small: ", ts_small)

# X = np.arange(36)[:, None]

# Xtr, Ytr, Xte, Yte, Xval, Yval, scaler = make_forecasting_dataset(X, horizon=5,
#                                                                   test_percent=0.2,
#                                                                   val_percent=0.3)


# Generate training and test datasets
Xtr, Ytr, Xte, Yte, scaler = make_forecasting_dataset(ts_small,
                                                      horizon=24, # forecast horizon of 24h ahead
                                                      test_percent = 0.1)


print("Xtr.shape: ", Xtr.shape)
print("Ytr.shape: ", Ytr.shape)
print("Xte.shape: ", Xte.shape)
print("Yte.shape: ", Yte.shape)
print("Xtr: ", Xtr)
print("Ytr: ", Ytr)
print("Xte: ", Xte)
print("Yte: ", Yte)
print("scaler: ", scaler)



print("Xtr: ", scaler.inverse_transform(Xtr.T)[0])
print("Ytr: ", scaler.inverse_transform(Ytr.T))
# print("Xval: ", scaler.inverse_transform(Xval.T)[0])
# print("Yval: ", Yval.T)
print("Xte: ", scaler.inverse_transform(Xte.T)[0])
print("Yte: ", Yte.T)



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


n_drop=10
states_tr = res.get_states(Xtr[None,:,:], n_drop=n_drop, bidir=False)
states_te = res.get_states(Xte[None,:,:], n_drop=n_drop, bidir=False)


print("states_tr.shape: ", states_tr.shape)
print("states_te.shape: ", states_te.shape)


pca = PCA(n_components=75)
states_tr = pca.fit_transform(states_tr[0])
states_te = pca.transform(states_te[0])

print("states_tr_pca.shape: ", states_tr.shape)
print("states_te_pca.shape: ", states_te.shape)


# Fit the ridge regression model
ridge = Ridge(alpha=1.0) 
ridge.fit(states_tr, Ytr[n_drop:,:])

# Compute the predictions
Yhat = ridge.predict(states_te)

print("Yhat: ", Yhat.shape)
print("Yhat: ", Yhat)


Yhat_transformed = scaler.inverse_transform([Yhat]).T
print("Yhat_transformed.shape: ", Yhat_transformed.shape)
print("Yhat_transformed: ", Yhat_transformed)
print("Yte: ", Yte.shape)
print("Yte: ", Yte)

fig = plt.figure(figsize=(14,4))
plt.plot(Yte[n_drop:,:], 'k--', label="True", linewidth=2)
plt.plot(Yhat_transformed, label="Predicted")
plt.grid()
plt.legend()
plt.title("True vs predicted electricity load")
# plt.show()
plt.savefig("predictions_ridge.png")


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
plt.plot(Yte[n_drop:,:], 'k--', label="True", linewidth=2)
plt.plot(scaler.inverse_transform(median_predictions[:,None]), label="Median prediction", color="tab:blue")
plt.fill_between(np.arange(len(Yte[n_drop:,:])), scaler.inverse_transform(percentile_5_predictions[:,None]).ravel(), scaler.inverse_transform(percentile_95_predictions[:,None]).ravel(), alpha=0.3, label="90% CI", color="tab:blue")
plt.grid()
plt.legend()
plt.title("Predicted electricity load using Gradient Boosting Regression Trees")
# plt.show()
plt.savefig("predictions_gbrt.png")

exit()

