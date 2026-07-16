import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, NHITS, RNN
from neuralforecast.losses.pytorch import MQLoss, QuantileLoss

# 1. Trading Parameters & Horizon Settings
TAKE_PROFIT_PCT = 0.004  # +0.4%
STOP_LOSS_PCT = 0.004    # -0.4%
HORIZON = 12             # 1 hour using 5-minute intervals (12 * 1)
LOOKBACK = 6            # .5 hours lookback window (12 * .5)


cols = [
    "sma_cross", "sma_cross_15m", "sma_cross_1h", "lowerband_r", "lowerband_r_15m", "lowerband_r_1h", "stochk_slope", "stochk_slope_15m", "stochk_slope_1h",
    "ha_slope_2", "ha_slope_2_15m", "ha_slope_2_1h", "ema_cross", "ema_cross_15m", "ema_cross_1h",
    "ha_slope_10", "ha_slope_10_15m", "ha_slope_10_1h",

    "ppo", "ppo_15m", "ppo_1h", "upperband_r", "upperband_r_15m", "upperband_r_1h", "sar_r", "sar_r_15m", "sar_r_1h",
    "kama_diff", "kama_diff_15m", "kama_diff_1h",
    # "macd", "macd_15m", "macd_1h", "macd_signal", "macd_signal_15m", "macd_signal_1h",
    "Close", "Open", "High", "Low",

    "di_plus", "di_plus_15m", "di_plus_1h", "di_minus", "di_minus_15m", "di_minus_1h", "cci_ha", "cci_ha_15m", "cci_ha_1h", "rsi", "rsi_15m", "rsi_1h", "rsi_ha", "rsi_ha_15m", "rsi_ha_1h", "di_diff", "di_diff_15m", "di_diff_1h", "willr", "willr_15m", "willr_1h",
    "hour_sin", "hour_cos"

]

df = pd.read_csv("X.csv", index_col=0, parse_dates=True)
df = df.iloc[30000:][cols]

df['unique_id'] = 'DAX40'

df.reset_index(drop=False, inplace=True)
df.rename(columns={'date_merge': 'ds'}, inplace=True)

df['y'] = df['Close'].shift(12)
df.dropna(inplace=True)

print(df)
# df_y = pd.read_csv("y_train.csv", index_col=0, parse_dates=True)
# df_y = df_y["labeling_multi"]

np.random.seed(42)

# Split into train and the very last point for prediction
train_df = df.iloc[:-1].copy()
last_known_actual = df["y"].iloc[-1]

# 3. Configure Probabilistic NHITS with Exogenous Variables
# We predict 9 quantiles from 10% to 90% to map out the price distribution
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

model = NHITS(
    input_size=LOOKBACK,
    h=HORIZON,
    loss=MQLoss(quantiles=quantiles),
    hist_exog_list=cols, # Features known only up to the present
    max_steps=150
)

nf = NeuralForecast(models=[model], freq="5min")
nf.fit(df=train_df)

# 4. Predict the Future Quantile Paths
# NeuralForecast automatically collects the required lookback window from train_df
forecast_df = nf.predict()

# 5. Calculate Absolute Price Targets
tp_target = last_known_actual * (1 + TAKE_PROFIT_PCT)
sl_target = last_known_actual * (1 - STOP_LOSS_PCT)

# 6. Extract Quantile Forecasts & Calculate Probabilities
# Columns will be named like 'NHITS-median', 'NHITS-quantile-0.1', etc.
quantile_cols = [c for c in forecast_df.columns if "quantile" in c or "median" in c]
num_quantiles = len(quantile_cols)

hit_tp_count = 0
hit_sl_count = 0
hit_flat_count = 0

# Analyze each quantile line as a potential future price path
for col in quantile_cols:
    predicted_path = forecast_df[col].values
    
    # Check if and where this specific quantile path crosses targets
    tp_indices = np.where(predicted_path >= tp_target)[0]
    sl_indices = np.where(predicted_path <= sl_target)[0]
    
    first_tp = tp_indices[0] if len(tp_indices) > 0 else float('inf')
    first_sl = sl_indices[0] if len(sl_indices) > 0 else float('inf')
    
    if first_tp == float('inf') and first_sl == float('inf'):
        hit_flat_count += 1
    elif first_tp < first_sl:
        hit_tp_count += 1
    else:
        hit_sl_count += 1

# Convert path outcomes into statistical probabilities
prob_tp = (hit_tp_count / num_quantiles) * 100
prob_sl = (hit_sl_count / num_quantiles) * 100
prob_flat = (hit_flat_count / num_quantiles) * 100

# 7. Print Results
print(f"Last Price: {last_known_actual:.2f}")
print(f"Targets -> TP: {tp_target:.2f} | SL: {sl_target:.2f}\n")
print(f"--- Probabilities over the next 2 hours ---")
print(f"Probability of hitting TP first: {prob_tp:.1f}%")
print(f"Probability of hitting SL first: {prob_sl:.1f}%")
print(f"Probability of remaining FLAT:   {prob_flat:.1f}%")
