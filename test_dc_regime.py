from src.data_utils.features import DC_market_regime
import pandas as pd
import os

df = pd.read_csv('data/raw/data_ohlc_features_1h.csv', parse_dates=True, index_col='date').iloc[10000:]
print(len(df))
df['dc_market_regime'] = DC_market_regime(df, col_close='Close', col_high='High', col_low='Low', threshold=0.003)        #40s

print(df['dc_market_regime'].head(100))

# df.to_csv('data/raw/data_ohlc_features_1h_dc_market_regime.csv', index=True)


dc_market_regime_roll = []
for i in df.rolling(window=1000,method='table'):
    dc_market_regime_roll.append(DC_market_regime(i, col_close="Close", col_high="High", col_low="Low", threshold=0.003)[-1])

dc_market_regime_roll = pd.Series(dc_market_regime_roll, name='dc_market_regime_roll', index=df.index)
print(dc_market_regime_roll.head(100))
df['dc_market_regime_roll'] = dc_market_regime_roll
df.to_csv('data/raw/data_ohlc_features_1h_dc_market_regime_test.csv', index=True)