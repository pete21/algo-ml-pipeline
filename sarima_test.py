import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

plt.style.use('seaborn-v0_8')

from src.data_utils.features import sarima_features, sarima_features_rolling_1_step

# ==========================================================
# 1. SETUP SIMULATED DATA
# ==========================================================
np.random.seed(42)
n_samples = 7200

# Create a clear weekly seasonal cycle (7 periods) with a slight upward trend
time_index = pd.date_range(start="2026-01-02", periods=n_samples, freq="5min")

print(time_index.shape)

trend = np.linspace(10, 20, n_samples)
seasonal_pattern = 5 * np.sin(2 * np.pi * np.arange(n_samples) / 7)
noise = np.random.normal(0, 1, n_samples)

series = pd.Series(trend + seasonal_pattern + noise, index=time_index)

# # Split into Train (80%) and Test (20%) datasets
# split_point = int(len(series) * 0.8)
# train_data = series.iloc[:split_point]
# test_data = series.iloc[split_point:]

series.plot(figsize=(12, 6))
plt.show()
plt.savefig('sarima_test.png')

series_df = pd.DataFrame(series, columns = ['series'])
series_df['local_date'] = series.index
dates = np.unique(series.index.date)


sarima_result_1_step = sarima_features_rolling_1_step(series_df, dates[1:], 2, 'series', 288)

sarima_result_full = sarima_features(series_df, dates[1:], 2, 'series', 288)

print(sarima_result_1_step.shape)
print(sarima_result_full.shape)

print(sarima_result_1_step)
print(sarima_result_full)


# # Core evaluation metrics
# mae = mean_absolute_error(test_data, predictions_series)
# mse = mean_squared_error(test_data, predictions_series)
# rmse = np.sqrt(mse)

# print("\n--- Out-of-Sample Performance Evaluation ---")
# print(f"Mean Absolute Error (MAE): {mae:.4f}")
# print(f"Mean Squared Error (MSE):   {mse:.4f}")
# print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# # Display a snapshot of the true values vs predictions vs errors
# eval_df = pd.DataFrame({
#     'Actual Value': test_data,
#     'SARIMA Prediction': predictions_series,
#     'Error Term (Residual)': test_residuals
# })
# print("\nFirst 5 rows of test data results:")
# print(eval_df.head())