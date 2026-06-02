#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install quantreo
#!pip install TA-Lib
#!pip install hurst
#!pip install feature-engine
#!pip install pykalman


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


pd.options.display.max_rows = 100

from features import *
from target import *

from FeaturesImportance import *


from datetime import datetime, date

from random import random, randint

# Import scikit-learn packages
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

plt.style.use("seaborn-v0_8")

from feature_engine.timeseries.forecasting import LagFeatures

import re


# # Data Import

# In[3]:


print(np.sqrt(0.8))
print(np.sqrt(0.6))
print(np.sqrt(0.4))
print(np.sqrt(0.2))
print(np.sqrt(0.05))


# In[4]:


print(np.sqrt(2))
print(np.sqrt(3))
print(np.sqrt(4))
print(np.sqrt(6))
print(np.sqrt(12))
print(np.sqrt(24))
print(np.sqrt(48))
print(np.sqrt(96))


# In[5]:


timeframes = ['10s', '1m', '2m', '3m', '4m', '5m', '10m', '15m', '20m', '30m', '1h', '2h', '4h', '1d']
timeframe_minutes = [0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 60, 120, 240, 1440]
timeframe_scalers = [0.22, 0.44, 0.63, 0.77, 0.89, 1, 1.4, 1.7, 2, 2.4, 3.4, 4.8, 6.8, 9.6]
path = '/home/jovyan/notebooks/freqtrade/user_data/data/binance/futures/'
dax_data = {}


# In[6]:


indexes = [1,5,7,10]
index_barrier = 1
index_base = 5
indexes_higher = [7,10]


# In[7]:


for i in indexes:
    file = 'DAX40_USDT_USDT-' + timeframes[i] + '-futures.parquet'     #DAX40_USDT_USDT-10s-futures.parquet
    print(file)
    # Import the data
    dax_data[i] = pd.read_parquet(path+file)
    dax_data[i].drop(columns=['volume'], inplace=True)
    dax_data[i]['date']=pd.to_datetime(dax_data[i]['date'], unit='ms', utc=True)
    dax_data[i].set_index('date', inplace=True)
    dax_data[i].rename(columns={'open':'Open','high':'High','low':'Low','close':'Close'}, inplace=True)
    dax_data[i] = dax_data[i].loc[dax_data[i].index.date>=date(2023,4,3)]
    dax_data[i] = dax_data[i].loc[~((dax_data[i].index.day==1) & (dax_data[i].index.month==1))]              # remove 1 Jan from data
    print(dax_data[i].head())


# In[8]:


dax_data[index_barrier].loc[:,'Close_wavelet'] = wavelet_denoising2(dax_data[index_barrier]['Close'], wavelet='db6', lvl=8, clear_levels=3)


# In[9]:


dax_data[index_barrier].iloc[53830:].head(20)


# In[10]:


dax_data[index_base].iloc[11030:].head(20)


# In[11]:


dax_data[7].iloc[3670:].head(20)


# In[12]:


for i in indexes_higher:
    dax_data[i]["date_merge"] = (
        dax_data[i].index
        + pd.to_timedelta(timeframe_minutes[i], "m")
        - pd.to_timedelta(timeframe_minutes[index_base], "m")
    )
    print(dax_data[i].head())


# ### Barrier high_time, low_time

# In[13]:


high_time_col='High'
low_time_col='Low'
high_time_col='Close_wavelet'
low_time_col='Close_wavelet'


# In[14]:


dax_data[index_base] = (dax_data[index_barrier].groupby(dax_data[index_barrier].index.floor(f'{timeframes[index_base]}in'))      #ceil
            .agg(Open=('Open','first'),
                 High=('High','max'),
                 Low=('Low','min'),
                 Close=('Close','last'),
                 Close_wavelet=('Close_wavelet','last'),
                 high_time=(high_time_col,'idxmax'),
                 low_time=(low_time_col,'idxmin')
                ))


# In[15]:


dax_data[index_base]["date_merge"] = dax_data[index_base].index


# In[16]:


dax_data[index_base]


# # Features Engineering

# ## Static features

# In[17]:


dax_data[index_base] = dax_data[index_base].iloc[:,0:8]


# In[18]:


dax_data[index_base]


# In[19]:


unique_dates = np.unique(dax_data[index_base].index.date)
unique_weekdates = []
for d in unique_dates:
    if d.weekday()<5:
        unique_weekdates.append(d)
print(len(unique_dates), len(unique_weekdates))

mondays_indexes = [i for i, n in enumerate(unique_dates) if n.weekday() == 0]
print(mondays_indexes)
num_mondays = sum(1 for i in unique_dates if i.weekday() == 0)
print(num_mondays)


# In[20]:




# In[20]:


static_features(dax_data[index_base], unique_weekdates, timeframe_scalers[index_base], high_col="High", low_col="Low", open_col="Open", close_col="Close")


# In[21]:


dax_data[index_base].isnull().sum().to_csv('nulls.csv')


# In[22]:


dax_data[index_base].isin([np.inf, -np.inf]).sum().to_csv('inf.csv')


# In[23]:


dax_data[index_base].columns.values


# In[24]:


dax_data[index_base].to_csv(f'dax_data_ohlc_features_{timeframes[index_base]}.csv')


# ## Static features higher timeframes

# In[25]:


for i in indexes_higher:
    print(f'Timeframe: {timeframes[i]}')
    static_features(dax_data[i], unique_weekdates, timeframe_scalers[i], high_col="High", low_col="Low", open_col="Open", close_col="Close")
    dax_data[i].to_csv(f'dax_data_ohlc_features_{timeframes[i]}.csv')


# ## Load static futures

# In[21]:


dax_data[index_base] = pd.read_csv(f'dax_data_ohlc_features_{timeframes[index_base]}.csv', parse_dates=True, index_col='date')
dax_data[index_base]["high_time"] = pd.to_datetime(dax_data[index_base]["high_time"])
dax_data[index_base]["low_time"] = pd.to_datetime(dax_data[index_base]["low_time"])
for i in indexes_higher:
    dax_data[i] = pd.read_csv(f'dax_data_ohlc_features_{timeframes[i]}.csv', parse_dates=True, index_col='date')
for i in (indexes_higher+[index_base]):
    dax_data[i]["date_merge"] = pd.to_datetime(dax_data[i]["date_merge"])


# ## Dynamic features

# In[22]:


parameters = {'n_estimators': 351, 'max_depth': 8, 'learning_rate': 0.02, 'subsample': 0.95, 'gamma': 0.9907448450762643, 'sma1_period': 6, 'sma2_period': 68, 'bb_periods': 34, 'bb_nbdev': 2.3215328600652314, 'ema1_period': 10, 'ema2_period': 12, 'sar_acc': 0.4818748156584099, 'sar_max': 1.0210305119144654, 'midprice_window': 2, 'l1_fast': 9, 'l2_fast': 6, 'l3_fast': 10, 'l1_slow': 12, 'l2_slow': 2, 'l3_slow': 22, 'kama_trend_period': 33, 'ha_candle_period': 7, 'dc_market_regime_period': 27, 'displacement_strength_period': 6, 'displacement_strength': 1.973639418953912, 'displacement_hull_period': 55, 'displacement_hull_slope_period': 18, 'gap_lookback': 7, 'gap_hull_period': 28, 'gap_hull_slope_period': 15, 'market_regime_threshold': 0.0028925039311096744, 'tenkan_window': 5, 'kijun_window': 34, 'cci_timeperiods': 30, 'macd_fastperiod': 15, 'macd_slowperiod': 39, 'macd_signalperiod': 5, 'price_distribution_window_size': 5, 'price_distribution_percentile_threshold': 0.2, 'rsi_period': 38, 'rsi_slope_period': 15, 'stoch_fastk_period': 10, 'stoch_slowk_period': 2, 'stoch_slowd_period': 18, 'ppo_fastperiod': 9, 'ppo_slowperiod': 43, 'stochrsi_timeperiod': 10, 'stochrsi_fastk_period': 7, 'stochrsi_fastd_period': 5, 'train_range_len': 58, 'test_range_len': 20, 'hour_range_start': 10, 'hour_range_stop': 20, 'adx_timeperiod': 5, 'di_timeperiod': 19, 'macd_slope_period': 9, 'sl': 0.0020838234550927073, 'tp': 0.003162728180601158, 'atr_period': 3, 'stochrsik_slope_period': 3, 'stochk_slope_period': 4, 'ha_sign_ma_period': 15, 'willr_timeperiod':14, 'target_tp': 0.0021515947383016956, 'ema_period': 15, 'ema_reversed_period': 5, 'threshold_long': 0.8923586168952897, 'threshold_short': 0.19967420435563193}
parameters = {'n_estimators': 382, 'max_depth': 7, 'learning_rate': 0.02, 'subsample': 0.95, 'gamma': 0.95, 'sma1_period': 5, 'sma2_period': 88, 'bb_periods': 18, 'bb_nbdev': 2.4625343800271473, 'ema1_period': 6, 'ema2_period': 19, 'sar_acc': 0.35347378805149954, 'sar_max': 1.102284705275549, 'midprice_window': 2, 'l1_fast': 8, 'l2_fast': 2, 'l3_fast': 8, 'l1_slow': 12, 'l2_slow': 3, 'l3_slow': 24, 'kama_trend_period': 25, 'ha_candle_period': 19, 'dc_market_regime_period': 19, 'displacement_strength_period': 35, 'displacement_strength': 1.0090179145664906, 'displacement_hull_period': 17, 'displacement_hull_slope_period': 14, 'gap_lookback': 6, 'gap_hull_period': 36, 'gap_hull_slope_period': 12, 'market_regime_threshold': 0.002987136210827556, 'tenkan_window': 7, 'kijun_window': 17, 'cci_timeperiods': 7, 'macd_fastperiod': 17, 'macd_slowperiod': 31, 'macd_signalperiod': 8, 'price_distribution_window_size': 5, 'price_distribution_percentile_threshold': 0.2, 'rsi_period': 27, 'rsi_slope_period': 6, 'stoch_fastk_period': 2, 'stoch_slowk_period': 4, 'stoch_slowd_period': 20, 'ppo_fastperiod': 3, 'ppo_slowperiod': 29, 'stochrsi_timeperiod': 7, 'stochrsi_fastk_period': 10, 'stochrsi_fastd_period': 13, 'train_range_len': 14, 'test_range_len': 5, 'hour_range_start': 10, 'hour_range_stop': 20, 'adx_timeperiod': 5, 'di_timeperiod': 19, 'macd_slope_period': 9, 'sl': 0.002294238807988796, 'tp': 0.002824953636382562, 'atr_period': 3, 'stochrsik_slope_period': 3, 'stochk_slope_period': 6, 'willr_timeperiod': 19, 'ha_sign_ma_period': 6, 'target_tp': 0.0024486199621331335, 'ema_period': 19, 'ema_reversed_period': 7, 'threshold_long': 0.8096038049250767, 'threshold_short': 0.17787209993585473}
p={}
p[7] = parameters
p[10] = parameters


# In[23]:


def calculate_features(df, parameters, scaler, col_close="close", col_high="high", col_low="low"):

    df = market_regime_features(df, col_close=col_close, col_high=col_high, col_low=col_low,
                         l1_fast=parameters['l1_fast'],l2_fast=parameters['l2_fast'],l3_fast=parameters['l3_fast'],
                         l1_slow=parameters['l1_slow'],l2_slow=parameters['l2_slow'],l3_slow=parameters['l3_slow'],
                         displacement_strength=parameters['displacement_strength'], market_regime_threshold=parameters['market_regime_threshold'],
                         price_distribution_window_size=parameters['price_distribution_window_size'],
                         price_distribution_percentile_threshold=parameters['price_distribution_percentile_threshold'],
                         kama_trend_period=parameters['kama_trend_period'],
                         ha_candle_period=parameters['ha_candle_period'], dc_market_regime_period=parameters['dc_market_regime_period'],
                         displacement_strength_period=parameters['displacement_strength_period'],
                         displacement_hull_period=parameters['displacement_hull_period'],
                         displacement_sma_period=20,
                         displacement_hull_slope_period=parameters['displacement_hull_slope_period'],
                         gap_lookback=parameters['gap_lookback'], gap_hull_period=parameters['gap_hull_period'], gap_hull_slope_period=parameters['gap_hull_slope_period'],
                         ha_sign_ma_period=parameters['ha_sign_ma_period']
                        )

    df['kama_trend_slow'] = df['kama_trend_slow']/scaler
    df['kama_trend_fast'] = df['kama_trend_fast']/scaler
    df['kama_trend_slow_diff'] = df['kama_trend_slow'].diff()
    df['kama_trend_fast_diff'] = df['kama_trend_fast'].diff()
    df['kama_trend_slow_diff2'] = df['kama_trend_slow_diff'].diff()
    df['kama_trend_fast_diff2'] = df['kama_trend_fast_diff'].diff()

    df['kama_diff'] = df['kama_diff']/scaler

    df['ema_ha_wickstrength'] = df['ema_ha_wickstrength']/scaler

    df.loc[:,'atr'] = talib.ATR(df[col_high], df[col_low], df[col_close], timeperiod=parameters['atr_period'])

    df.loc[:,'upperband'], df.loc[:,'middleband'], df.loc[:,'lowerband'], df.loc[:,'ema1'], df.loc[:,'ema2'], df.loc[:,'sma1'], df.loc[:,'sma2'], df.loc[:,'midprice'], df.loc[:,'sar'] \
        = overlap(df, col_close=col_close, col_high=col_high, col_low=col_low, bb_periods=parameters['bb_periods'], bb_nbdev=parameters['bb_nbdev'], ema1_period=parameters['ema1_period'], sma1_period=parameters['sma1_period'], sma2_period=parameters['sma2_period'], sar_acc=parameters['sar_acc'], sar_max=parameters['sar_max'], midprice_window=parameters['midprice_window'])

    df.loc[:,'sma_cross'] = np.log(df['sma1']/df['sma2'])*100/scaler
    df.loc[:,'ema_cross'] = np.log(df['ema1']/df['ema2'])*100/scaler
    df.loc[:,'upperband_r'] = np.log(df['upperband']/df['ha_close'])*100/scaler
    df.loc[:,'middleband_r'] = np.log(df['middleband']/df['ha_close'])*100/scaler
    df.loc[:,'lowerband_r'] = np.log(df['lowerband']/df['ha_close'])*100/scaler
    df.loc[:,'sar_r'] = np.log(df['sar']/df['ha_close'])*100/scaler

    df.loc[:,'tenkan_sen'], df.loc[:,'kijun_sen'] = ichimoku(df, col_high="ha_high", col_low="ha_low", tenkan_window=parameters['tenkan_window'], kijun_window=parameters['kijun_window'])

    df.loc[:,'r_tenkan_sen'] = np.log(df['tenkan_sen']/df['ha_close'])*100/scaler
    df.loc[:,'r_kijun_sen'] = np.log(df['kijun_sen']/df['ha_close'])*100/scaler

    # momentum

    df.loc[:,'cci'] = talib.CCI(df[col_high], df[col_low], df[col_close], timeperiod=parameters['cci_timeperiods'])/100
    df.loc[:,'cci_ha'] = talib.CCI(df['ha_high'], df['ha_low'], df['ha_close'], timeperiod=parameters['cci_timeperiods'])/100


    df.loc[:,'macd'], df.loc[:,'macdsignal'], df.loc[:,'macdhist'] = talib.MACDEXT(df[col_close],
                                                                 fastperiod=parameters['macd_fastperiod'], fastmatype=0,
                                                                 slowperiod=parameters['macd_slowperiod'], slowmatype=0,
                                                                 signalperiod=parameters['macd_signalperiod'], signalmatype=0)
    df.loc[:,'macd_slope'] = talib.LINEARREG_ANGLE(df['macd'], parameters['macd_slope_period'])
    df['macdhist'] = df['macdhist']/scaler

    df.loc[:,'rsi'] = (talib.RSI(df[col_close], timeperiod=parameters['rsi_period'])-50)/100
    df.loc[:,'rsi_ha'] = (talib.RSI(df['ha_close'], timeperiod=parameters['rsi_period'])-50)/100
    df.loc[:,'rsi_slope'] = talib.LINEARREG_ANGLE(df['rsi'], parameters['rsi_slope_period'])

    df.loc[:,'stochk'], df.loc[:,'stochd'] = talib.STOCH(df[col_high], df[col_low], df[col_close], fastk_period=parameters['stoch_fastk_period'],
                                           slowk_period=parameters['stoch_slowk_period'], slowk_matype=0,
                                           slowd_period=parameters['stoch_slowd_period'], slowd_matype=0)
    df.loc[:,'stochk'] = (df['stochk']-50)/100
    df.loc[:,'stochd'] = (df['stochd']-50)/100

    df.loc[:,'stochk_slope'] = talib.LINEARREG_ANGLE(df['stochk'], parameters['stochk_slope_period'])

    df.loc[:,'stochrsik'], df.loc[:,'stochrsid'] = talib.STOCHRSI(df[col_close], timeperiod=parameters['stochrsi_timeperiod'],
                                              fastk_period=parameters['stochrsi_fastk_period'], fastd_period=parameters['stochrsi_fastd_period'], fastd_matype=0)
    df.loc[:,'stochrsik'] = (df['stochrsik']-50)/100
    df.loc[:,'stochrsid'] = (df['stochrsid']-50)/100

    df.loc[:,'stochrsik_slope'] = talib.LINEARREG_ANGLE(df['stochrsik'], parameters['stochrsik_slope_period'])


    df.loc[:,'ppo'] = talib.PPO(df['ha_close'], fastperiod=parameters['ppo_fastperiod'], slowperiod=parameters['ppo_slowperiod'], matype=0)/scaler

    df.loc[:,'willr'] = talib.WILLR(df.High, df.Low, df.Close, timeperiod=parameters['willr_timeperiod'])/50+1

    # Compute Hilbert Transform Dominant Cycle
    #df['Hilbert_Dominant_Cycle'] = hilbert_dominant_cycle(df[col_close])

    df.loc[:,'adx'] = talib.ADX(df[col_high],df[col_low],df[col_close],timeperiod=parameters['adx_timeperiod'])
    df.loc[:,'di_plus'] = talib.PLUS_DI(df[col_high],df[col_low],df[col_close],timeperiod=parameters['di_timeperiod'])/10
    df.loc[:,'di_minus'] = talib.MINUS_DI(df[col_high],df[col_low],df[col_close],timeperiod=parameters['di_timeperiod'])/10
    df.loc[:,'di_diff'] = df['di_plus'] - df['di_minus']

    # # Prepare data
    # wavelet_coeff = wavelet_transform(df['ha_close'], 8)
    # df.loc[:,'wavelet_reconstr'] = inverse_wavelet_transform(wavelet_coeff, 8, 4)

    return df


# In[ ]:





# In[24]:


dax_data[index_base] = calculate_features(dax_data[index_base], parameters, timeframe_scalers[index_base], col_close="Close", col_high="High", col_low="Low")


# In[25]:


dax_data[index_base].columns


# In[26]:


#dax_data[index_base][dax_data[index_base].select_dtypes(np.float16).columns] = dax_data[index_base].select_dtypes(np.float16).astype(np.float32)
#dax_data[index_base][dax_data[index_base].select_dtypes(np.int64).columns] = dax_data[index_base].select_dtypes(np.int64).astype(np.int32)
dax_data[index_base].dtypes


# In[27]:


for i in indexes_higher:
    dax_data[i] = calculate_features(dax_data[i], p[i], timeframe_scalers[i], col_close="Close", col_high="High", col_low="Low")
#    dax_data[i][dax_data[i].select_dtypes(np.float16).columns] = dax_data[i].select_dtypes(np.float16).astype(np.float32)
#    dax_data[i][dax_data[i].select_dtypes(np.int64).columns] = dax_data[i].select_dtypes(np.int64).astype(np.int32)


# In[ ]:





# ## Plots

# In[30]:


plt.figure(figsize=(14, 10))
plt.plot(dax_data[index_base]['ha_close'], label='ha_close')

plt.show()


# In[31]:


plt.figure(figsize=(14, 10))
plt.plot(dax_data[index_base].loc[:,['corr_2','corr_4','corr_6']].iloc[4000:4200,])

plt.show()


# In[32]:


dfc = dax_data[index_base].iloc[ 4000:4400, :]

fig, ax1 = plt.subplots(figsize=(12, 8))

color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('price', color=color)
ax1.plot(dfc.index, dfc['ha_close'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('target', color=color)  # we already handled the x-label with ax1
# ax2.plot(dfc.index, dfc[["rogers_satchell_vol_2"]], color='blue')
# ax2.plot(dfc.index, dfc[["rogers_satchell_vol_4"]], color='green')
ax2.plot(dfc.index, dfc[["rogers_satchell_vol_8"]], color='black')
ax2.plot(dfc.index, dfc[["rogers_satchell_vol_12"]], color='grey')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()



# In[33]:


dfc = dax_data[index_base].iloc[ 4000:4600, :]

fig, ax1 = plt.subplots(figsize=(12, 8))

color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('price', color=color)
ax1.plot(dfc.index, dfc['ha_close'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('target', color=color)  # we already handled the x-label with ax1
# ax2.plot(dfc.index, dfc[["tail_index_1"]], color='green')
ax2.plot(dfc.index, dfc[["close_regr_entropy"]], color='green')
ax2.plot(dfc.index, dfc[["permutation_entropy"]], color='black')
# ax2.plot(dfc.index, dfc[["labeling_dual_ema"]], color='grey')
# ax2.plot(dfc.index, dfc[["labeling_multi"]], color='brown')

ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()



# In[34]:


# Plot pca
vol_symbol='rogers_satchell_vol_'
plt.figure(figsize=(15,6))
# plt.plot(dax_data[index_base].loc[:,["vol_short_pca1"]].iloc[4000:4200,], label="vol_short_pca1")
# plt.plot(dax_data[index_base].loc[:,["vol_long_pca1"]].iloc[4000:4200,], label="vol_long_pca1")
plt.plot(dax_data[index_base].loc[:,[vol_symbol + "2"]].iloc[4000:4200,], label=vol_symbol + "2", alpha=0.5)
plt.plot(dax_data[index_base].loc[:,[vol_symbol + "4"]].iloc[4000:4200,], label=vol_symbol + "4", alpha=0.5)
# plt.plot(dax_data[index_base].loc[:,[vol_symbol + "8"]].iloc[4000:4200,], label=vol_symbol + "8", alpha=0.5)
# plt.plot(dax_data[index_base].loc[:,[vol_symbol + "16"]].iloc[4000:4200,], label=vol_symbol + "16", alpha=0.5)
plt.legend()
plt.show()


# In[35]:


dfc = dax_data[index_base].iloc[ 4040:4140, :]

fig, ax1 = plt.subplots(figsize=(12, 8))

color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('price', color=color)
ax1.plot(dfc.index, dfc['ha_close'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('target', color=color)  # we already handled the x-label with ax1
ax2.plot(dfc.index, dfc[["ema_ha_sign"]], color='blue')
# ax2.plot(dfc.index, dfc[["candle_filling"]], color='green')
ax2.plot(dfc.index, dfc[["ema_ha_wickstrength"]], color='pink')
# ax2.plot(dfc.index, dfc[["ha_candle_fill"]], color='yellow')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

# candle_sign, candle_filling, body_amplitude, ha_wickstrength, ha_sign, ha_candle_fill


# In[36]:


dfc = dax_data[index_base].iloc[ 4000:4200, :]

fig, ax1 = plt.subplots(figsize=(12, 8))

color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('price', color=color)
ax1.plot(dfc.index, dfc['Close'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('target', color=color)  # we already handled the x-label with ax1
ax2.plot(dfc.index, dfc[["log_ret_1"]])
ax2.plot(dfc.index, dfc[["log_ret_2"]])
ax2.plot(dfc.index, dfc[["log_ret_3"]])
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# In[37]:


dfc = dax_data[index_base].iloc[ 4000:4200, :]

fig, ax1 = plt.subplots(figsize=(12, 8))

color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('price', color=color)
ax1.plot(dfc.index, dfc['Close'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('target', color=color)  # we already handled the x-label with ax1
ax2.plot(dfc.index, dfc[["adf_stat"]], color=color)
ax2.plot(dfc.index, dfc[["adf_pvalue"]], color='green')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# In[38]:


plt.figure(figsize=(14, 10))
plt.plot(dax_data[index_base].loc[:,['Close', 'kama_regime_slow','kama_regime_fast' ]].iloc[5300:5600,])

plt.show()


# In[39]:


dfc = dax_data[10].iloc[5300:5600, :]

fig, ax1 = plt.subplots(figsize=(12, 8))

color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('price', color=color)
ax1.plot(dfc.index, dfc['Close'], color=color)
#ax1.plot(dfc.index, dfc['kama_trend'], color=color)

ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('target', color=color)  # we already handled the x-label with ax1
ax2.plot(dfc.index, dfc["kama_trend_slow_diff"], color=color)
ax2.plot(dfc.index, dfc[["kama_diff"]], color='green')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# In[40]:


plt.figure(figsize=(14, 10))
plt.plot(dax_data[index_base].loc[:,['Close','bullish_gap_low','bullish_gap_high','bearish_gap_low','bearish_gap_high']].iloc[54000:54400,])
plt.ylim(16000,17000)
plt.show()


# In[41]:


dfc = dax_data[index_base].iloc[ 14400:14500, :]
timeperiod=4
dfc['gap_ema'] = talib.WMA(dfc.loc[:,"bullish_gap"]+dfc.loc[:,"bearish_gap"], timeperiod)
dfc['gap_ema_slope'] = talib.LINEARREG_ANGLE(dfc['gap_ema'], 9)
fig, ax1 = plt.subplots(figsize=(12, 8))

color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('price', color=color)
ax1.plot(dfc.index, dfc['Close'], color='black')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('target', color=color)  # we already handled the x-label with ax1
#ax2.plot(dfc.index, dfc["bullish_gap"]+dfc["bearish_gap"])
#ax2.plot(dfc.index, dfc[["bearish_gap"]])
#ax2.plot(dfc.index, dfc["gap_hull"])
ax2.plot(dfc.index, dfc["gap_ema_slope"], color='green')
ax2.plot(dfc.index, dfc["gap_hull_slope"], color='grey')
#ax2.plot(dfc.index, (dfc["bullish_gap_size"].rolling(window=40).mean())-(dfc["bearish_gap_size"].rolling(window=40).mean()))
#ax2.plot(dfc.index, (dfc["bullish_gap_size"].ewm(span=45, adjust=False).mean())-(dfc["bearish_gap_size"].ewm(span=45, adjust=False).mean()))
#ax2.plot(dfc.index, (dfc["bullish_gap_size"].rolling(window=1).mean()))
#ax2.plot(dfc.index, (dfc["bearish_gap_size"].rolling(window=1).mean()))
#ax2.plot(dfc.index, (dfc["bullish_gap"]+dfc["bearish_gap"]).rolling(window=40).mean())
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# In[42]:


dfc = dax_data[5].iloc[ 2400:2500, :]

fig, ax1 = plt.subplots(figsize=(12, 8))

color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('price', color=color)
ax1.plot(dfc.index, dfc['Close'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('target', color=color)  # we already handled the x-label with ax1
#ax2.plot(dfc.index, dfc[["displacement"]].rolling(window=25).mean(), color=color)
ax2.plot(dfc.index, dfc[["displacement"]].rolling(window=30).mean(), color=color)
ax2.plot(dfc.index, dfc[["displacement_hull"]], color='green')
ax2.plot(dfc.index, dfc[["displacement_hull_slope"]], color='grey')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# In[43]:


dfc = dax_data[index_base].iloc[ 2400:2500, :]
timeperiod=4
dfc['displacement_ema'] = talib.EMA(dfc["displacement"], timeperiod)
dfc['displacement_ema_slope'] = talib.LINEARREG_ANGLE(dfc['displacement_ema'], 9)
fig, ax1 = plt.subplots(figsize=(12, 8))

color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('price', color=color)
ax1.plot(dfc.index, dfc['Close'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('target', color=color)  # we already handled the x-label with ax1
ax2.plot(dfc.index, dfc[["displacement_ema"]], color='black')
ax2.plot(dfc.index, dfc[["displacement_ema_slope"]], color='green')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# In[44]:


plt.figure(figsize=(14, 10))
plt.plot(dax_data[index_base].loc[:,[ 'up_displacement_high','down_displacement_low','Close']].iloc[10000:10500,])
plt.ylim(15800,16200)
plt.show()


# In[45]:


# Prepare the stacked layers
# bottom_25 = dax_data[index_base]["0_to_25"].iloc[1200:1500,]
# middle = dax_data[index_base]["25_to_75"].iloc[1200:1500,]
# top = dax_data[index_base]["75_to_100"].iloc[1200:1500,]

# # Create the stacked area plot
# fig, ax = plt.subplots(figsize=(15, 6))

# ax.fill_between(bottom_25.index, 0, bottom_25, color='blue', alpha=0.5, label='0–25%')
# ax.fill_between(middle.index, bottom_25, bottom_25 + middle, color='orange', alpha=0.5, label='25–75%')
# ax.fill_between(top.index, bottom_25 + middle, bottom_25 + middle + top, color='green', alpha=0.5, label='75–100%')

# # Customize the plot
# ax.set_ylim(0, 100)
# ax.set_title("Close Price Distribution within Range Zones", fontsize=14)
# ax.set_xlabel("Date")
# ax.set_ylabel("Percentage of Closes in Zone")
# ax.legend()
# ax.grid(True)

# plt.tight_layout()
# plt.show()


# In[46]:


dfc = dax_data[10].iloc[ 3420:3600, :]

fig, ax1 = plt.subplots(figsize=(12, 8))

color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('price', color=color)
ax1.plot(dfc.index, dfc['Close'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('target', color=color)  # we already handled the x-label with ax1
#ax2.plot(dfc.index, dfc[["displacement"]].rolling(window=25).mean(), color=color)
#ax2.plot(dfc.index, dfc[["sine"]], color=color)
#ax2.plot(dfc.index, dfc[["leadsine"]], color='green')
ax2.plot(dfc.index, dfc[["sine_slope_10"]], color=color)
ax2.plot(dfc.index, dfc[["sine_diff_slope_10"]], color='pink')
#ax2.plot(dfc.index, dfc[["ha_slope_5"]], color='green')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# In[47]:


# Take the data
dfc = dax_data[index_base].iloc[4000:4250]

# Extract DC and Trend events
dc_events_up, dc_events_down, dc_events = calculate_dc(dfc, col_close='Close', col_high='High', col_low='Low', threshold=0.0015)
#trend_events_down, trend_events_up = calculate_trend(dfc, dc_events_down, dc_events_up)


min_price = dfc['Close'].min()
max_price = dfc['Close'].max()

plt.figure(figsize=(12, 8))
plt.plot(dfc['Close'], label='Price')

# Add DC and OS events to the plot
for start, end in dc_events_up:
    plt.fill_between(dfc.index[start:end+1], min_price, max_price, alpha=0.3, color='green', label='DC Event')

for start, end in dc_events_down:
    plt.fill_between(dfc.index[start:end+1], min_price, max_price, alpha=0.3, color='red', label='DC Event')


#for start, end in trend_events_up:
#    plt.fill_between(dfc.index[start:end+1], min_price, max_price, alpha=0.1, color='green', label='DC Event')
#    
#for start, end in trend_events_down:
#    plt.fill_between(dfc.index[start:end+1], min_price, max_price, alpha=0.1, color='red', label='DC Event')

plt.show()


# In[48]:


print(dc_events_up)
print(dc_events_down)


# In[49]:


dfc = dax_data[index_base].iloc[4000:4250, :]

fig, ax1 = plt.subplots(figsize=(12, 8))

color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('price', color=color)
ax1.plot(dfc.index, dfc['Close'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('target', color=color)  # we already handled the x-label with ax1
#ax2.plot(dfc.index, dfc[["displacement"]].rolling(window=25).mean(), color=color)
#ax2.plot(dfc.index, dfc[["dc_market_regime"]].rolling(window=30).mean(), color='blue)
#ax2.plot(dfc.index, dfc["dc_market_regime_ema"], color='green')
#ax2.plot(dfc.index, dfc["dc_market_regime_wma"], color='blue')
log_wma = np.log1p(np.abs(dfc["dc_market_regime_wma"]))*np.sign(dfc["dc_market_regime_wma"])
log_ema = np.log1p(np.abs(dfc["dc_market_regime_ema"]))*np.sign(dfc["dc_market_regime_ema"])
ax2.plot(dfc.index, log_wma, color='grey')
ax2.plot(dfc.index, log_ema, color='green')
#ax2.plot(dfc.index, (log_wma+log_ema)/2, color='black')
#ax2.plot(dfc.index, (dfc["dc_market_regime_wma"]+dfc["dc_market_regime_ema"])/2, color='pink')
#ax2.plot(dfc.index, (dfc["dc_market_regime_ema"]+log_wma)/2, color='pink')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# In[50]:


dfc = dax_data[index_base].iloc[4000:4050, :]

fig, ax1 = plt.subplots(figsize=(12, 8))

color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('price', color=color)
ax1.plot(dfc.index, dfc['Close'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('target', color=color)  # we already handled the x-label with ax1
ax2.plot(dfc.index, dfc["long_pivot"]-dfc["short_pivot"], color='green')
# ax2.plot(dfc.index, dfc["short_pivot"], color='blue')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# # Target

# In[51]:


#parameters_target = {'target_tp': 0.004, 'ema_period': 7, 'ema_reversed_period': 20, 'threshold_long': 0.85, 'threshold_short': 0.15}
#target_tp - tolerancja zmiennosci, im wieksza wartosc, tym mniejsza tolerancja


# In[28]:


dax_data[index_base].loc[:,"labeling_binary"], dax_data[index_base].loc[:,"labeling_dual_ema"], dax_data[index_base].loc[:,"labeling_multi"] = build_target(dax_data[index_base], \
            close_col="Close", high_col="High", low_col="Low", high_time_col="high_time", \
            low_time_col="low_time", tp=parameters['target_tp'], ema_period=parameters['ema_period'], ema_reversed_period=parameters['ema_reversed_period'], \
            threshold_long=parameters['threshold_long'], threshold_short=parameters['threshold_short'])


# In[36]:


dax_data[index_base].loc[:,"labeling_binary"], dax_data[index_base].loc[:,"labeling_dual_ema"], dax_data[index_base].loc[:,"labeling_multi"] = build_target(dax_data[index_base], \
            close_col="Close_wavelet", high_col="Close_wavelet", low_col="Close_wavelet", high_time_col="high_time", \
            low_time_col="low_time", tp=parameters['target_tp'], ema_period=30, ema_reversed_period=15, \
            threshold_long=0.9, threshold_short=0.1)


# In[29]:


# Print actual value count
print(f"Value counts for each class:\n{dax_data[index_base].labeling_multi.value_counts()}\n")

# Display pie chart to visually check the proportion
dax_data[index_base].loc[:,"labeling_multi"].value_counts().plot.pie(y='label', title='Proportion of each class')
plt.show()


# In[30]:


# Take the data
#dfc = dax_data[index_base].loc[ (dax_data[index_base].index>=pd.to_datetime(unique_dates[1757])) & (dax_data[index_base].index<pd.to_datetime(unique_dates[1759])), :]

dfc = dax_data[index_base].iloc[ 31400:31550, :]

fig, ax1 = plt.subplots(figsize=(12, 8))

color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('price', color=color)
ax1.plot(dfc.index, dfc['Close'], color=color)
ax1.plot(dfc.index, dfc['Close_wavelet'], color='violet')

ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('target', color=color)  # we already handled the x-label with ax1
ax2.plot(dfc.index, dfc[["labeling_dual_ema"]], color=color)
ax2.plot(dfc.index, dfc[["labeling_multi"]], color='green')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# # Analysis

# In[31]:


list_X = [
'ha_close',
#'ha_low',
# 'High',
# 'Low',
'Close',
#'midprice',
# 'Open',
#'ha_high',
#'macdsignal',
#'ha_open',
#'sar',
'tenkan_sen',
#'ema1',
#'sma1',
'kijun_sen',
#'lowerband',
'sine',
#'middleband',
#'upperband',
'sine_diff_slope_2',
#'ema2',
#'sma2',
'sine_diff_slope_5',
#'kama_trend_slow',
#'kama_trend_fast',
'kama_trend_slow_diff',
'kama_trend_fast_diff',
'kama_trend_slow_diff2',
'kama_trend_fast_diff2',
'macdhist',
'sma_cross',
#'log_ret_1',
'lowerband_r',
#'down_displacement_low',
#'down_displacement_high',
#'bullish_gap_size',
#'dc_market_regime',
#'displacement',
#'bearish_gap_size',
#'macd',
'stochk_slope',
#'dc_market_regime_ema',
#'dc_market_regime_wma',
'dc_market_regime_ema_log',
'ha_slope_2',
'ema_cross',
'log_ret_ha_1',
#'log_ret_2',
'ppo',
'upperband_r',
'sar_r',
'log_ret_ha_2',
#'ha_slope_15',
'stochd',
'kama_diff',
#'bullish_gap_low',
#'bullish_gap_high',
'bearish_gap',
#'log_ret_3',
'rsi_slope',
'sine_diff',
'stochrsid',
#'bearish_gap_low',
#'bearish_gap_high',
'stochrsik',
'bullish_gap',
'log_ret_ha_3',
#'ha_slope_5',
'ha_slope_10',
#'log_ret_4',
'r_kijun_sen',
#'ema_ha_upper_wick',
#'log_ret_ha_4',
#'displacement_sma',
'r_tenkan_sen',
#'ha_sign',
#'log_ret_ha_5',
#'log_ret_5',
#'middleband_r',
#'ema_ha_lower_wick',
'displacement_hull',
'displacement_hull_slope',
'stochk',
'ema_ha_sign',
'ema_ha_wickstrength',
'gap_hull',
'gap_hull_slope',
'di_plus',
'di_minus',
'cci_ha',
#'cci',
'rsi',
'rsi_ha',
'di_diff',
'willr',
#'vol_short_pca1',
#'vol_long_pca1',
#'wavelet_reconstr',
#'hurst',
#'hurst_kalman',
'close_wavelet_rolling',
# 'close_regr_entropy',
'permutation_entropy',
"tail_index_1",
"skew",
"petrosian_fd",

'long_pivot',
'short_pivot'
]


# In[32]:


col_y=['labeling_multi']


# In[33]:


def getXy(data, index_b, indexes_h, parameters, p, scalers, X_cols, y_col: str, cutoff_date: date, col_open="Open", col_high="High", col_low="Low", col_close="Close"):
    cutoff_date_prev_day = cutoff_date - pd.Timedelta(3, "D")
    ml_data = {}
    ml_data[index_b] = calculate_features(data[index_b], parameters, scalers[index_b], col_close=col_close, col_high=col_high, col_low=col_low)
    ml_data[index_b] = ml_data[index_b][list_X + [y_col] + [col_open, col_high, col_low, "date_merge"]].loc[ml_data[index_b].index.date>=cutoff_date_prev_day]

    target = ml_data[index_b].loc[(ml_data[index_b].index.hour>=parameters['hour_range_start']) & (ml_data[index_b].index.hour<=parameters['hour_range_stop']), [y_col]]

    lag_f = LagFeatures(variables = list_X + [col_open, col_high, col_low], periods=[1,2], drop_na=True)

    for i in indexes_h:
        ml_data[i] = calculate_features(data[i], p[i], scalers[i], col_close=col_close, col_high=col_high, col_low=col_low)
        ml_data[i] = ml_data[i][list_X + [col_open, col_high, col_low, "date_merge"]].loc[ml_data[i].index.date>=cutoff_date_prev_day]
        # print(ml_data[i].columns.values)
        ml_data[i] = lag_f.fit_transform(ml_data[i]).add_suffix(f"_{timeframes[i]}")
        # print(ml_data[i].columns.values)

    # print(ml_data[index_b].columns.values)
    ml_data[index_b] = lag_f.fit_transform(ml_data[index_b])
    ml_data[index_b] = ml_data[index_b].loc[(ml_data[index_b].index.hour>=parameters['hour_range_start']) & (ml_data[index_b].index.hour<=parameters['hour_range_stop'])]
    # print(ml_data[index_b].columns.values)

    for i in indexes_h:
#        ml_data[i] = ml_data[i].loc[(ml_data[i][f"date_merge_{timeframes[i]}"].dt.hour>=parameters['hour_range_start']) & (ml_data[i][f"date_merge_{timeframes[i]}"].dt.hour<=parameters['hour_range_stop'])]

#        ml_data[i] = ml_data[i].loc[(ml_data[i].index.hour>=parameters['hour_range_start']) & (ml_data[i].index.hour<=parameters['hour_range_stop'])]
#        ml_data[index_b] = ml_data[index_b].merge(ml_data[i], how='left', left_index=True, right_index=True)

        ml_data[index_b] = pd.merge_ordered(
            ml_data[index_b],
            ml_data[i],
            fill_method="ffill",
            left_on="date_merge",
            right_on=f"date_merge_{timeframes[i]}",
            how="left"
        )
    ml_data[index_b].rename(columns={"date_merge": "date"}, inplace=True)
    ml_data[index_b].set_index('date', inplace=True, drop=True)
    ml_data[index_b] = ml_data[index_b].loc[ml_data[index_b].index.date>=cutoff_date]
    # print(ml_data[index_b].columns.values)

#    ml_data[index_b][ml_data[index_b].select_dtypes(np.float16).columns] = ml_data[index_b].select_dtypes(np.float16).astype(np.float32)
    ml_data[index_b][ml_data[index_b].select_dtypes(np.float64).columns] = ml_data[index_b].select_dtypes(np.float64).astype(np.float32)
    ml_data[index_b][ml_data[index_b].select_dtypes(np.int64).columns] = ml_data[index_b].select_dtypes(np.int64).astype(np.int32)

    # feature_columns = lag_f.get_feature_names_out()
    # feature_columns.remove(y_col)
    # feature_columns.remove("date_merge")
    feature_columns = [x for x in lag_f.get_feature_names_out() if x not in ([y_col] + ["date_merge"])]           # [col_open, col_high, col_low, col_close] - exclude open, high, low, close

    X_columns = []
    for x in feature_columns:
        X_columns.append(x)
        for i in indexes_h:

#             m_search = re.search('(?:slope|log)_(\d+)', x, flags=re.ASCII)
#             if m_search:
#                 m=m_search.group(1)
#                 if timeframe_minutes[i]*int(m) > 240:
# #                    print(f'Skipped: {x},{timeframes[i]},lag {m}')
#                     continue

            X_columns.append(f"{x}_{timeframes[i]}")

    dates = np.unique(ml_data[index_b].index.date)
# PCA
    # print('log_ret_ha_short_pca')                      # ['log_ret_ha_short_pca1','log_ret_ha_short_pca2']
    # cols = [x for x in X_columns if x.startswith("ret_ha_log") and (x[-1].isdigit() or x.endswith("15m"))]
    # pca_res = calc_kernel_pca(ml_data[index_b], dates[1:], 10, cols, ['log_ret_ha_short_pca1','log_ret_ha_short_pca2'])
    # ml_data[index_b]=ml_data[index_b].join(pca_res)
    # X_columns.append('log_ret_ha_short_pca1')
    # X_columns.append('log_ret_ha_short_pca2')
# X_columns = [x for x in X_columns if x not in cols]

    # print('log_ret_ha_long_pca')
    # cols = [x for x in X_columns if x.startswith("ret_ha_log") and (x.endswith("1h"))]
    # pca_res = calc_kernel_pca(ml_data[index_b], dates[1:], 10, cols, ['log_ret_ha_long_pca1','log_ret_ha_long_pca2'])
    # ml_data[index_b]=ml_data[index_b].join(pca_res)
    # X_columns.append('log_ret_ha_long_pca1')
    # X_columns.append('log_ret_ha_long_pca2')
# X_columns = [x for x in X_columns if x not in cols]

    print('ichimoku_short_pca')
    cols = [x for x in X_columns if (x.startswith("tenkan_sen") or x.startswith("kijun_sen")) and not(x.endswith("1h"))]
    pca_res = calc_kernel_pca(ml_data[index_b], dates[1:], 10, cols, ['ichimoku_short_pca1','ichimoku_short_pca2'])
    ml_data[index_b]=ml_data[index_b].join(pca_res)
    X_columns.append('ichimoku_short_pca1')
    X_columns.append('ichimoku_short_pca2')
#    X_columns = [x for x in X_columns if x not in cols]

    print('ichimoku_long_pca')
    cols = [x for x in X_columns if (x.startswith("tenkan_sen") or x.startswith("kijun_sen")) and (x.endswith("1h"))]
    pca_res = calc_kernel_pca(ml_data[index_b], dates[1:], 10, cols, ['ichimoku_long_pca1','ichimoku_long_pca2'])
    ml_data[index_b]=ml_data[index_b].join(pca_res)
    X_columns.append('ichimoku_long_pca1')
    X_columns.append('ichimoku_long_pca2')
#    X_columns = [x for x in X_columns if x not in cols]

    print('kama_short_pca')
    cols = [x for x in X_columns if (x.startswith("kama_trend_slow_diff") or x.startswith("kama_trend_fast_diff")) and not (x.endswith("1h"))]
    pca_res = calc_kernel_pca(ml_data[index_b], dates[1:], 10, cols, ['kama_short_pca1','kama_short_pca2'])
    ml_data[index_b]=ml_data[index_b].join(pca_res)
    X_columns.append('kama_short_pca1')
    X_columns.append('kama_short_pca2')
#    X_columns = [x for x in X_columns if x not in cols]

    print('kama_long_pca')
    cols = [x for x in X_columns if (x.startswith("kama_trend_slow_diff") or x.startswith("kama_trend_fast_diff")) and (x.endswith("1h"))]
    pca_res = calc_kernel_pca(ml_data[index_b], dates[1:], 10, cols, ['kama_long_pca1','kama_long_pca2'])
    ml_data[index_b]=ml_data[index_b].join(pca_res)
    X_columns.append('kama_long_pca1')
    X_columns.append('kama_long_pca2')
#    X_columns = [x for x in X_columns if x not in cols]

    #print(X_columns)
    #print(len(X_columns))
    cols=[
        # 'sine','sine_lag_1','sine_lag_2',
        # 'sine_15m','sine_lag_1_15m','sine_lag_2_15m',
        # 'lowerband_r_1h','lowerband_r_lag_1_1h','lowerband_r_lag_2_1h',
        # 'upperband_r_1h','upperband_r_lag_1_1h','upperband_r_lag_2_1h',
        # 'ema_ha_wickstrength_15m','ema_ha_wickstrength_lag_1_15m','ema_ha_wickstrength_lag_2_15m',
        # 'stochrsid_1h','stochrsid_lag_1_1h','stochrsid_lag_2_1h',
        # 'rsi_1h','rsi_lag_1_1h','rsi_lag_2_1h',
        # 'rsi_ha_1h','rsi_ha_lag_1_1h','rsi_ha_lag_2_1h',
        # 'rsi','rsi_lag_1','rsi_lag_2',
        'kama_trend_slow_diff','kama_trend_slow_diff_15m','kama_trend_slow_diff_1h',
        'kama_trend_fast_diff','kama_trend_fast_diff_15m','kama_trend_fast_diff_1h',
        'kama_trend_slow_diff2','kama_trend_slow_diff2_15m','kama_trend_slow_diff2_1h',
        'kama_trend_fast_diff2','kama_trend_fast_diff2_15m','kama_trend_fast_diff2_1h',
        'kama_trend_slow_diff_lag_1','kama_trend_slow_diff_lag_1_15m','kama_trend_slow_diff_lag_1_1h',
        'kama_trend_fast_diff_lag_1','kama_trend_fast_diff_lag_1_15m','kama_trend_fast_diff_lag_1_1h',
        'kama_trend_slow_diff2_lag_1','kama_trend_slow_diff2_lag_1_15m','kama_trend_slow_diff2_lag_1_1h',
        'kama_trend_fast_diff2_lag_1','kama_trend_fast_diff2_lag_1_15m','kama_trend_fast_diff2_lag_1_1h',
        'kama_trend_slow_diff_lag_2','kama_trend_slow_diff_lag_2_15m','kama_trend_slow_diff_lag_2_1h',
        'kama_trend_fast_diff_lag_2','kama_trend_fast_diff_lag_2_15m','kama_trend_fast_diff_lag_2_1h',
        'kama_trend_slow_diff2_lag_2','kama_trend_slow_diff2_lag_2_15m','kama_trend_slow_diff2_lag_2_1h',
        'kama_trend_fast_diff2_lag_2','kama_trend_fast_diff2_lag_2_15m','kama_trend_fast_diff2_lag_2_1h',

        'tenkan_sen','tenkan_sen_lag_1','tenkan_sen_lag_2',
        'tenkan_sen_15m','tenkan_sen_lag_1_15m','tenkan_sen_lag_2_15m',
        'tenkan_sen_1h','tenkan_sen_lag_1_1h','tenkan_sen_lag_2_1h',
        'kijun_sen','kijun_sen_lag_1','kijun_sen_lag_2',
        'kijun_sen_15m','kijun_sen_lag_1_15m','kijun_sen_lag_2_15m',
        'kijun_sen_1h','kijun_sen_lag_1_1h','kijun_sen_lag_2_1h',
    ]
    X_columns = [x for x in X_columns if x not in cols]
    # print(X_columns)
    X = ml_data[index_b][X_columns]
    y = ml_data[index_b][y_col]
    #print(lag_f.get_feature_names_out())
    X.fillna(0,inplace=True)
    return X, y, X_columns


# In[34]:


cols = [x for x in dax_data[index_base].columns if (x.startswith("kama_trend_slow_diff") or x.startswith("kama_trend_fast_diff")) and not(x.endswith("1h"))]
print(cols)


# In[95]:


X, y, X_columns = getXy(dax_data, index_base, indexes_higher, parameters, p, timeframe_scalers, list_X, col_y[0], date(2025,1,1), col_open="Open", col_high="High", col_low="Low", col_close="Close")


# In[97]:


dax_data[index_base].isnull().sum().values


# In[98]:


dax_data[index_base].isin([np.inf, -np.inf]).sum().values


# ## Entropy

# In[99]:


# Example with sample entropy on returns
dax_data[index_base]["sample_entropy_returns"] = fe.math.sample_entropy(df=dax_data[index_base], col='log_ret_ha_1', window_size=50)


# In[100]:


plt.figure(figsize=(14, 10))
plt.plot(dax_data[index_base].loc[:,['sample_entropy_returns']].iloc[59000:59700,])

plt.show()


# ## Correlation

# In[101]:


dax_data[index_base].describe()


# In[102]:


col_y


# In[103]:


correlations=[]
correlations.append(dax_data[index_base].drop(columns=['high_time','low_time','date_merge']).dropna().corr()[col_y])


# In[104]:


correlation = correlations[0]
#correlation[:-4]


# In[105]:


correlation[:-4].abs().sum().sum()


# In[106]:


pd.set_option('display.max_rows', None)
correlation[:].unstack()


# In[107]:


plt.figure(figsize=(10, 30))
sns.heatmap(correlation[:-4], annot=True, fmt=".2f", cmap='viridis', linewidths=.5)
plt.title("Correlation on the Train set with the features and the target", size=15)
plt.show()


# ## Features Importance

# In[64]:


from FeaturesImportance import *


# In[65]:


cols = dax_data[index_base].columns
remove = ['Open', 'High', 'Low', 'Close', 'high_time', 'low_time', 'labeling_binary', 'labeling_dual_ema', 'labeling_multi', 'Close_wavelet', 'date_merge', ]
cols = [x for x in cols if x not in remove]
cols


# In[76]:


feature_importance(dax_data[index_base].iloc[10000:20000].loc[:,cols+col_y], cols, col_y[0], reg=False, mda=False)


# ## Multicollinearity detection using VIF

# In[108]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

X, y, X_columns = getXy(dax_data, index_base, indexes_higher, parameters, p, timeframe_scalers, list_X, col_y[0], date(2024,6,1), col_open="Open", col_high="High", col_low="Low", col_close="Close")

#df_copy=dax_data[index_base][list_X].iloc[130000:150000].dropna()
#lag_f = LagFeatures(variables = list_X, periods=[1,2], drop_na=True)
#df_copy = lag_f.fit_transform(df_copy)
#X = df_copy[lag_f.get_feature_names_out()]      #df_copy[list_X]

# X.drop(columns=[
#     'tenkan_sen','tenkan_sen_lag_1','tenkan_sen_lag_2',
#     'tenkan_sen_15m','tenkan_sen_lag_1_15m','tenkan_sen_lag_2_15m',
#     'tenkan_sen_1h','tenkan_sen_lag_1_1h','tenkan_sen_lag_2_1h'], inplace=True)
# X.drop(columns=[
#     'kijun_sen','kijun_sen_lag_1','kijun_sen_lag_2',
#     'kijun_sen_15m','kijun_sen_lag_1_15m','kijun_sen_lag_2_15m',
#     'kijun_sen_1h','kijun_sen_lag_1_1h','kijun_sen_lag_2_1h'], inplace=True)
# X.drop(columns=[
#     'kama_trend_slow_diff','kama_trend_slow_diff_15m','kama_trend_slow_diff_1h',
#     'kama_trend_fast_diff','kama_trend_fast_diff_15m','kama_trend_fast_diff_1h',
#     'kama_trend_slow_diff2','kama_trend_slow_diff2_15m','kama_trend_slow_diff2_1h',
#     'kama_trend_fast_diff2','kama_trend_fast_diff2_15m','kama_trend_fast_diff2_1h',
#     'kama_trend_slow_diff_lag_1','kama_trend_slow_diff_lag_1_15m','kama_trend_slow_diff_lag_1_1h',
#     'kama_trend_fast_diff_lag_1','kama_trend_fast_diff_lag_1_15m','kama_trend_fast_diff_lag_1_1h',
#     'kama_trend_slow_diff2_lag_1','kama_trend_slow_diff2_lag_1_15m','kama_trend_slow_diff2_lag_1_1h',
#     'kama_trend_fast_diff2_lag_1','kama_trend_fast_diff2_lag_1_15m','kama_trend_fast_diff2_lag_1_1h',
#     'kama_trend_slow_diff_lag_2','kama_trend_slow_diff_lag_2_15m','kama_trend_slow_diff_lag_2_1h',
#     'kama_trend_fast_diff_lag_2','kama_trend_fast_diff_lag_2_15m','kama_trend_fast_diff_lag_2_1h',
#     'kama_trend_slow_diff2_lag_2','kama_trend_slow_diff2_lag_2_15m','kama_trend_slow_diff2_lag_2_1h',
#     'kama_trend_fast_diff2_lag_2','kama_trend_fast_diff2_lag_2_15m','kama_trend_fast_diff2_lag_2_1h'], inplace=True)
# X.drop(columns=[
#     'kama_trend_slow_diff_1h','kama_trend_fast_diff_1h',
#     'kama_trend_slow_diff2_1h','kama_trend_fast_diff2_1h',
#     'kama_trend_slow_diff_lag_1_1h','kama_trend_fast_diff_lag_1_1h',
#     'kama_trend_slow_diff2_lag_1_1h','kama_trend_fast_diff2_lag_1_1h',
#     'kama_trend_slow_diff_lag_2_1h','kama_trend_fast_diff_lag_2_1h',
#     'kama_trend_slow_diff2_lag_2_1h','kama_trend_fast_diff2_lag_2_1h'], inplace=True)
# X.drop(columns=[
#     'log_ret_ha_1','log_ret_ha_3',
#     'log_ret_ha_1_lag_1','log_ret_ha_3_lag_1',
#     'log_ret_ha_1_lag_2','log_ret_ha_3_lag_2',
#     'log_ret_ha_1_15m','log_ret_ha_3_15m',
#     'log_ret_ha_1_lag_1_15m','log_ret_ha_3_lag_1_15m',
#     'log_ret_ha_1_lag_2_15m','log_ret_ha_3_lag_2_15m',], inplace=True)
# X.drop(columns=[
#     'log_ret_ha_1_1h','log_ret_ha_3_1h',
#     'log_ret_ha_1_lag_1_1h','log_ret_ha_3_lag_1_1h',
#     'log_ret_ha_1_lag_2_1h','log_ret_ha_3_lag_2_1h',], inplace=True)


print('vif')
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)


# In[109]:


vif_data.sort_values(by='VIF').to_csv('vif.csv')


# # Random Forest Classifier

# In[110]:


#X, y, X_columns = getXy(dax_data, index_base, indexes_higher, parameters, p, timeframe_scalers, list_X, col_y[0], date(2024,6,1), col_open="Open", col_high="High", col_low="Low", col_close="Close")

# Split the dataset into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

train_split = int(len(X.index) * random())
print(train_split)

X_train, X_test = X.iloc[np.maximum(0,train_split-10000):train_split], X.iloc[train_split: np.minimum(train_split+10000,len(X.index))]
y_train, y_test = y.iloc[np.maximum(0,train_split-10000):train_split], y.iloc[train_split: np.minimum(train_split+10000,len(X.index))]

# Create and train a RandomForest model
model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)


# In[112]:


y_pred = model.predict(X_test)


# In[113]:


y_pred


# In[114]:


y_test.values


# In[115]:


score = np.multiply(y_pred, y_test.values).sum()
print(score)
penalty = 0
#for i,j,k in zip(y_pred,y_test,target.iloc[train_split: np.minimum(train_split+10000,len(X.index))]):
#    if i!=0 and j==0:
#        penalty += (i-k)*i
print(penalty)
#score -=penalty/10


# In[116]:


plt.figure(figsize=(15,6))
plt.plot(y_test.values[1000:2000], label="TRUE")
plt.plot(y_pred[1000:2000], label="TEST")
plt.legend()


# In[117]:


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, classification_report, roc_auc_score
# Calcul de la matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Calcul de l'accuracy
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}")

prec = precision_score(y_test, y_pred, average=None)
print(f"\nPrecision: {prec}")

# Calcul du score F1
f1 = f1_score(y_test, y_pred, average=None)
print(f"F1 Score: {f1}")


# In[118]:


conf_matrix[0][0]+conf_matrix[2][2]-conf_matrix[0][2]-conf_matrix[2][0]-0.5*(conf_matrix[1][0]+conf_matrix[1][2])


# In[119]:


print('roc_auc score:')
pred_prob = model.predict_proba(X_test)
roc_auc_score(y_test.values, pred_prob, multi_class='ovr')


# In[120]:


ax= plt.subplot()

sns.heatmap(conf_matrix, annot=True ,cmap="YlGn", ax=ax)

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.xaxis.set_ticklabels(['sell', 'out', 'buy']); ax.yaxis.set_ticklabels(['sell', 'out', 'buy']);
plt.show()


# # Robustness

# In[121]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

def plot_timeseries_cv_indices(cv, X, y, ax, n_splits, lw=5):
    """Visualize results of TimeSeriesSplit."""

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X, y)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the training and test sets
        ax.scatter(range(len(indices)), [ii + 0.5] * len(indices),
                   c=indices, marker="_", lw=lw, cmap='coolwarm_r', vmin=-0.2, vmax=1.2)

    # Visualize the targets
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y, marker="_", lw=lw*6, cmap='RdYlGn', vmin=0, vmax=1)

    # Format the plot
    yticklabels = list(range(n_splits)) + ['target']
    ax.set(yticks=np.arange(n_splits + 1) + 0.5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel='CV iteration', 
           ylim=[n_splits + 1.6, -0.1], xlim=[0, len(X)])
    ax.set_title('TimeSeriesSplit', fontsize=15)
    return ax


# In[122]:


# Create the TimeSeriesSplit object
n_splits = 10
tscv = TimeSeriesSplit(n_splits=n_splits)

# Visualize the results
fig, ax = plt.subplots(figsize=(15, 8))
plot_timeseries_cv_indices(tscv, X, y.values, ax, n_splits)
plt.tight_layout()
plt.show()


# In[134]:


from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, TimeSeriesSplit, cross_val_predict

#df_copy = dax_data[index_base][list_X + col_y + ['Open','High','Low','Close']].iloc[10000:60000]
#df_copy = df_copy[~np.isinf(df_copy).any(axis=1)]

# Create feature matrix X and target vector y
#X = df_copy[list_X].iloc[:-1,:].values
#y = df_copy[col_y].iloc[1:].values
#X = df_copy[list_X + ['Open','High','Low','Close']]
#y = df_copy[col_y[0]]


# Initialize classifier
clf = RandomForestClassifier(random_state=42)

# K-Fold cross validation
tscv = TimeSeriesSplit(n_splits=n_splits)
#kf = KFold(n_splits=n_splits, shuffle=False)

# Collect scores
scores = cross_val_score(clf, X, y, cv=tscv, scoring="accuracy")

print(f"Scores for each fold: {scores}")
print(f"Average Precision: {np.mean(scores):.2f}")
print(f"Standard Deviation: {np.std(scores):.2f}")


# In[135]:


scores


# In[ ]:


#y_pred = cross_val_predict(clf, X, y, cv=tscv)
#conf_mat = confusion_matrix(y, y_pred)
#print(conf_mat)


# In[ ]:





# ### WARNING
# Very good accuracy but BE CAREFUL, the acccuracy doesn't mean anything for us. We need to know the HIT ratio when we enter in position

# # PCA

# In[ ]:


# Import the Features Engineering Package from Quantreo
import quantreo.features_engineering as fe

# Import scikit-learn packages
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

# Import polars
import polars as pd

# To display the graphics
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")


# In[ ]:


daxeur = dax_data[index_base].iloc[100000:105000]


# In[ ]:


#vol_features = [col for col in daxeur.columns if "vol_" in col and "volatility" not in col]
vol_features = ['rogers_satchell_vol_3','rogers_satchell_vol_6','rogers_satchell_vol_12',
                           'yang_zhang_vol_3','yang_zhang_vol_6','yang_zhang_vol_12']


# In[ ]:


daxeur[vol_features].plot(figsize=(15,6))
plt.show()


# In[ ]:


# Define the train size
train_size = int(len(daxeur) * 0.8)

# Chronological split
train_df = daxeur.iloc[:train_size].copy()
test_df = daxeur.iloc[train_size:].copy()

# Check the result
print(f"Train set shape: {train_df.shape}")
print(f"Test set shape : {test_df.shape}")


# In[ ]:


# Volatility Features Correlation
train_df[vol_features].corr()


# In[ ]:


# Standardize the features using only the training set
scaler = StandardScaler()
scaler.fit(train_df[vol_features])  # Fit on training data only

# Transform both train and test dataset using the same scaler
train_df_scaled = scaler.transform(train_df[vol_features])
test_df_scaled = scaler.transform(test_df[vol_features])

# Convert the scaled test dataset to a DataFrame for easier handling
test_df_scaled = pd.DataFrame(test_df_scaled, index=test_df.index, columns=vol_features)

train_df_scaled = pd.DataFrame(train_df_scaled, index=train_df.index, columns=vol_features)


# Display the standardized volatility features
test_df_scaled


# In[ ]:


# Select columns with 'float64' dtype  
float64_cols = list(df_vol_scaled.select_dtypes(include='float64'))

# The same code again calling the columns
df_vol_scaled[float64_cols] = df_vol_scaled[float64_cols].astype('float32')


# In[ ]:


# Call the PCA method from scikit learn
pca = KernelPCA(n_components=1, kernel='rbf')

# Train the PCA on the train set
train_pca_scores = pca.fit_transform(train_df_scaled)

# Apply the PCA on the test dataset
test_pca_scores = pca.transform(test_df_scaled)
test_pca_scores.shape


# In[ ]:


train_pca_scores_df = pd.DataFrame(train_pca_scores, columns = ['vol_pca1', 'vol_pca2'], index=train_df.index)
train_pca_scores_df


# In[ ]:


test_pca_scores_df = pd.DataFrame(test_pca_scores, columns = ['vol_pca1'], index=test_df.index)
test_pca_scores_df


# In[ ]:


# Plot the result
plt.figure(figsize=(15,6))
plt.plot(test_pca_scores_df["vol_pca1"], label="PCA1")
#plt.plot(train_pca_scores_df["pca2"], label="PCA2")
plt.plot(test_df_scaled["rogers_satchell_vol_3"], label="rogers_satchell_vol_3", alpha=0.5)
plt.plot(test_df_scaled["rogers_satchell_vol_6"], label="rogers_satchell_vol_6", alpha=0.5)
plt.plot(test_df_scaled["rogers_satchell_vol_12"], label="rogers_satchell_vol_12", alpha=0.5)
#plt.plot(train_df_scaled["parkinson_vol_60"], label="parkinson_vol_60", alpha=0.5)
plt.legend()
plt.show()


# # Optuna

# In[ ]:


#!pip install xgboost
#!pip install cupy-cuda12x


# In[41]:


from pandas.plotting import scatter_matrix, autocorrelation_plot
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, precision_score, f1_score
from sklearn.metrics import log_loss, cohen_kappa_score, hinge_loss
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE
from sklearn.neural_network import MLPClassifier

import math
#from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#from statsmodels.tsa.arima_model import ARIMA
from xgboost import XGBClassifier
import xgboost as xgb
import lightgbm as lgb
from lightgbm import LGBMClassifier


#import cupy as cp

import optuna
from optuna.visualization import plot_param_importances, plot_contour, plot_slice
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_rank
from optuna.visualization import plot_slice
from optuna.visualization import plot_timeline
from optuna.importance import get_param_importances


# In[42]:


range_start = 30050
range_stop = 32000

fig, ax1 = plt.subplots(figsize=(12, 8))

color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('price', color=color)
ax1.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['Close'].iloc[range_start:range_stop], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('target', color=color)  # we already handled the x-label with ax1

ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['labeling_multi'].iloc[range_start:range_stop], color=color)
#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['lowerband_r'].iloc[range_start:range_stop], color='blue')
#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['upperband_r'].iloc[range_start:range_stop], color='green')
#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['sma_cross'].iloc[range_start:range_stop], color='green')
#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['sma1'].iloc[range_start:range_stop], color='blue')
#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['sma2'].iloc[range_start:range_stop], color='pink')
#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['ema_cross'].iloc[range_start:range_stop], color='blue')
#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['midprice'].iloc[range_start:range_stop], color='blue')
#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['sar'].iloc[range_start:range_stop], color='green')
#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['tenkan_sen_r'].iloc[range_start:range_stop], color='blue')
#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['kijun_sen_r'].iloc[range_start:range_stop], color='pink')

#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['cci'].iloc[range_start:range_stop], color='green')
#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['macd'].iloc[range_start:range_stop], color='blue')
#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['macdsignal'].iloc[range_start:range_stop], color='pink')
#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['macdhist'].iloc[range_start:range_stop], color='purple')

#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['rsi'].iloc[range_start:range_stop], color='green')
#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['stochrsik'].iloc[range_start:range_stop], color='blue')
#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['stochrsid'].iloc[range_start:range_stop], color='purple')

#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['stochk'].iloc[range_start:range_stop], color='blue')
#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['stochd'].iloc[range_start:range_stop], color='purple')

#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['ppo'].iloc[range_start:range_stop], color='green')

#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['log_ret_ha_20'].iloc[range_start:range_stop], color='blue')
#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['corr_20'].iloc[range_start:range_stop], color='purple')

#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['velocity'].iloc[range_start:range_stop], color='blue')
#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['acceleration'].iloc[range_start:range_stop], color='purple')

#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['vol_pca1'].iloc[range_start:range_stop], color='blue')
#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['vol_pca2'].iloc[range_start:range_stop], color='purple')

#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['adf_pvalue'].iloc[range_start:range_stop], color='blue')
#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['dc_market_regime_ema'].iloc[range_start:range_stop], color='purple')

#'kama_regime_fast', 'kama_regime_slow', 'kama_diff',       'kama_trend_ema',

#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['kama_regime_fast'].iloc[range_start:range_stop], color='blue')
#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['kama_regime_slow'].iloc[range_start:range_stop], color='purple')

#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['kama_diff'].iloc[range_start:range_stop], color='blue')
#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['kama_trend_ema'].iloc[range_start:range_stop], color='purple')

#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['candle_range_std'].iloc[range_start:range_stop], color='blue')
#ax2.plot(dax_data[index_base].iloc[range_start:range_stop].index, dax_data[index_base]['displacement_hull'].iloc[range_start:range_stop], color='purple')


ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# # Evaluate multiple models

# In[125]:


X, y, X_columns = getXy(dax_data, index_base, indexes_higher, parameters, p, timeframe_scalers, list_X, col_y[0], date(2024,6,1), col_open="Open", col_high="High", col_low="Low", col_close="Close")
y=y+1

train_split=int(len(X.index) * random())
#train_split=5000
train_start_idx = max(train_split-20000, 0)
test_end_idx = min(train_split+10000, len(X.index))

print(train_start_idx, train_split, test_end_idx)

X_train, X_test = X.iloc[train_start_idx:train_split], X.iloc[train_split: test_end_idx]
y_train, y_test = y[train_start_idx:train_split], y[train_split: test_end_idx]

sample_weights = compute_sample_weight(
    class_weight='balanced',
    y=y_train
)

# Calculate weights
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))


# In[126]:


X.columns


# In[127]:


sample_weights


# In[128]:


#X.loc[X.index.date>=date(2025,10,1)].to_csv('X3.csv')


# In[133]:


print('Observations: %d' % (len(X.index)))
print('X Training Observations: %d' % (len(X_train.index)))
print('X Testing Observations: %d' % (len(X_test.index)))
print('y Training Observations: %d' % (len(y_train)))
print('y Testing Observations: %d' % (len(y_test)))
num_folds = 2
scoring = 'accuracy'


# In[ ]:


# Source - https://datascience.stackexchange.com/q/80421

regressors = [["Linear Regression", LinearRegression()],
              ["Lasso Regression", Lasso()],
              ["Gaussian Process Regressor", GaussianProcessRegressor()],              
              ["SVR linear", SVR(kernel = 'linear', gamma='scale', max_iter = 1500)],
              ["SVR poly 2", SVR(kernel = 'poly', degree=2, gamma='scale', max_iter = 1500)],
              ["SVR poly 3", SVR(kernel = 'poly', degree=3, gamma='scale', max_iter = 1500)],
              ["SVR poly 4", SVR(kernel = 'poly', degree=4, gamma='scale', max_iter = 1500)],
              ["SVR poly 5", SVR(kernel = 'poly', degree=5, gamma='scale', max_iter = 1500)],
              ["SVR rbf C=0.01", SVR(kernel = 'rbf', C=0.01, gamma='scale', max_iter = 1500)],              
              ["SVR rbf C=0.1", SVR(kernel = 'rbf', C=0.1, gamma='scale', max_iter = 1500)],
              ["SVR rbf C=0.5", SVR(kernel = 'rbf', C=0.5, gamma='scale', max_iter = 1500)],
              ["SVR rbf C=1", SVR(kernel = 'rbf', C=1, gamma='scale', max_iter = 1500)],              
              ["SVR rbf C=10", SVR(kernel = 'rbf', C=10.0, gamma='scale', max_iter = 1500)],
              ["SVR rbf C=20", SVR(kernel = 'rbf', C=20.0, gamma='scale', max_iter = 1500)],
              ["SVR rbf C=50", SVR(kernel = 'rbf', C=50.0, gamma='scale', max_iter = 1500)],              
              ["SVR sigmoid", SVR(kernel = 'sigmoid', gamma='scale', max_iter = 1500)],
              ["GradientBoostingRegressor", GradientBoostingRegressor()],
              ["RandomForestRegressor", RandomForestRegressor(n_estimators = 150)],
              ["DecisionTreeRegressor", DecisionTreeRegressor(max_depth=10)],
              ["Bagging Regressor TREE", BaggingRegressor(base_estimator = DecisionTreeRegressor(max_depth=15))],
              ["Bagging Regressor FOREST", BaggingRegressor(base_estimator = RandomForestRegressor(n_estimators = 100))],
              ["Bagging Regressor linear", BaggingRegressor(base_estimator = LinearRegression(normalize=True))],
              ["Bagging Regressor lasso", BaggingRegressor(base_estimator = Lasso(normalize=True))],
              ["Bagging Regressor SVR rbf", BaggingRegressor(base_estimator = SVR(kernel = 'rbf', C=10.0, gamma='scale'))],
              ["Extra Trees Regressor", ExtraTreesRegressor(n_estimators = 150)],
              ["K-Neighbors Regressor 1", KNeighborsRegressor(n_neighbors=1)],
              ["K-Neighbors Regressor 2", KNeighborsRegressor(n_neighbors=2)],
              ["K-Neighbors Regressor 3", KNeighborsRegressor(n_neighbors=3)],
              ["AdaBoostRegressor", AdaBoostRegressor(base_estimator=None)],
              ["AdaBoostRegressor tree", AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=15))],
              ["AdaBoostRegressor forest", AdaBoostRegressor(base_estimator=RandomForestRegressor(n_estimators = 100))],
              ["AdaBoostRegressor lin reg", AdaBoostRegressor(base_estimator=LinearRegression(normalize=True))],
              ["AdaBoostRegressor lasso", AdaBoostRegressor(base_estimator = Lasso(normalize=True))]]


for reg in regressors:

     try:

           scores = cross_val_score(reg[1], X, y, cv=5)
           scores = np.average(scores)
           print('cross val score', scores)
           print()

     except:
          continue


# In[134]:


models = []
#models.append(('LR' , LogisticRegression(solver='saga', max_iter=4000)))
#
#models.append(('MLP' , MLPClassifier(hidden_layer_sizes = (200,20), max_iter=1000, activation='relu', solver='adam')))
#models.append(('LSVC' , LinearSVC(C=0.001)))

#models.append(('LDA' , LinearDiscriminantAnalysis()))
#models.append(('KNN' , KNeighborsClassifier(n_neighbors=3)))
#models.append(('CART' , DecisionTreeClassifier()))
#models.append(('NB' , GaussianNB()))
#models.append(('SVC' , SVC()))
#models.append(('SGDClassifier', SGDClassifier()))
#models.append(('RF' , RandomForestClassifier(n_estimators=250, max_depth=5)))
models.append(('LightGBM', LGBMClassifier(num_class=3, # device='cuda',
                           learning_rate=parameters['learning_rate'],
                           n_estimators=parameters['n_estimators'],
                           max_depth=parameters['max_depth'],
                           subsample=parameters['subsample'])))
models.append(('XGBoost', XGBClassifier(num_class=3, device='cuda',
                           learning_rate=parameters['learning_rate'],
                           n_estimators=parameters['n_estimators'],
                           max_depth=parameters['max_depth'],
                           subsample=parameters['subsample'],
                           gamma=parameters['gamma'])))


# In[135]:


# Evaluate each algorithm for accuracy
results = []
names = []

tss = TimeSeriesSplit(n_splits=num_folds)

for name, model in models[:]:
    print('Model: ' + name + ' Start: ' + datetime.now().strftime("%H:%M:%S"))

    # kfold = KFold(n_splits=num_folds)
    # Split & print out results
    # for i, (train_index, test_index) in enumerate(tss.split(X_train)):
    #     print(f"Fold {i+1}:")
    #     print(f"  train:{X[train_index]}")
    #     print(f"  test:{X[test_index]}")

    #   cv_results = cross_val_score(model, rescaled_X, y, cv=kfold, scoring=scoring)
    cv = cross_validate(model, X_train, y_train, cv=tss, scoring='balanced_accuracy', n_jobs=-1)

    results.append(cv)
    names.append(name)
    # msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(f'Model: {name} Score:')
    print(cv)
    print('Model: ' + name + ' End: ' + datetime.now().strftime("%H:%M:%S"))


#for name, model in models[:]:
    # print('Model: ' + name + ' Start: ' + datetime.now().strftime("%H:%M:%S"))
    # model.fit(X_train, y_train, sample_weight=sample_weights)
#   clf.fit(rescaled_X_train, y_train)

#    importances = clf.feature_importances_
    # Sort feature importances in descending order
#    indices = np.argsort(importances)[::-1]
    '''
        # Rearrange feature names so they match the sorted feature importances
        names = [X.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(X.shape[1]), importances[indices])
        plt.xticks(range(X.shape[1]), names, rotation=90)
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.show()
    '''

#     y_pred = model.predict(X_test)
# #   y_pred = clf.predict(rescaled_X_test)

#     # Calcul de la matrice de confusion
#     conf_matrix = confusion_matrix(y_test, y_pred)
#     print("Confusion Matrix:")
#     print(conf_matrix)
#     print(f"Confusion matrix score: {conf_matrix[0][0]+conf_matrix[2][2]-conf_matrix[0][2]-conf_matrix[2][0]-0.5*(conf_matrix[1][0]+conf_matrix[1][2])}")

#     # # Calcul de l'accuracy
#     # acc = accuracy_score(y_test, y_pred)
#     # print(f"\nAccuracy: {acc:.4f}")

#     # prec = precision_score(y_test, y_pred, average=None)
#     # print(f"\nPrecision: {prec}")

#     # # Calcul du score F1
#     # f1 = f1_score(y_test, y_pred, average=None)
#     # print(f"F1 Score: {f1}")


#     y_pred = (y_pred-1).flatten()
#     y_test_arr = (y_test.values-1).flatten()
#     #print(y_test_arr.shape)
#     #print(y_pred.shape)
#     score = y_pred * y_test_arr
#     print(f"My score: {score.sum()}")
#     print(name + ' End: ' + datetime.now().strftime("%H:%M:%S"))

#     print("Classification report:")
#     print(classification_report(y_test_arr, y_pred))

    # print("Predict probabilities:")
    # probability = model.predict_proba(X_test)
    # print(probability)
    # print("Model coefficients:")
    # coefficients=pd.DataFrame(zip(X.columns, np.transpose(clf.coef_)))
    # print(coefficients)


# ## Feature importance

# In[ ]:


# Coefficients and Odds Ratios
coefficients = model.coef_[0]
odds_ratios = np.exp(coefficients)


# Display feature importance using coefficients and odds ratios
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients,
    'Odds Ratio': odds_ratios
})
print("\nFeature Importance (Coefficient and Odds Ratio):")
print(feature_importance.sort_values(by='Coefficient', ascending=False))
feature_importance.sort_values(by='Coefficient', ascending=False).to_csv('feature_importance_odds.csv')


# In[ ]:


# Permutation Importance
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10000, random_state=42, n_jobs=-1)
perm_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance Mean': perm_importance.importances_mean,
    'Importance Std': perm_importance.importances_std
})
print("\nPermutation Importance:")
print(perm_importance_df.sort_values(by='Importance Mean', ascending=False))
perm_importance_df.sort_values(by='Importance Mean', ascending=False).to_csv('perm_importance_mean.csv')


# In[ ]:


perm_importance_df.sort_values(by='Importance Mean', ascending=False).to_csv('perm_importance_mean.csv')


# In[ ]:


# Recursive Feature Elimination (RFE)
rfe_model = LogisticRegression(max_iter=2000, solver='saga')
rfe = RFE(rfe_model, n_features_to_select=200)
rfe.fit(X_train, y_train)


rfe_features = X.columns[rfe.support_]
print("\nSelected Features by RFE:")
print(rfe_features)
rfe_features.to_csv('rfe_features.csv')


# In[ ]:


pd.DataFrame(rfe_features).to_csv('rfe_features.csv')


# In[ ]:


len(y_pred)


# In[ ]:


range_start = 2000
range_stop = 2500

fig, ax1 = plt.subplots(figsize=(12, 8))

color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('price', color=color)
#ax1.plot(X_test.iloc[range_start:range_stop].index, X_test.iloc[range_start:range_stop]['macd_slope'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('target', color=color)  # we already handled the x-label with ax1
ax2.plot(X_test.iloc[range_start:range_stop].index, y_pred[range_start:range_stop], color='blue')
ax2.plot(X_test.iloc[range_start:range_stop].index, y_test_arr[range_start:range_stop], color='green')
#ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# # Backtesting

# In[35]:


from backtesting import Backtest, Strategy
from backtesting.lib import SignalStrategy
import multiprocessing as mp
#mp.set_start_method('spawn', force=True)


# In[39]:


'''
for i,j in zip(y_pred, ml_data['Close'].loc[(y.index>train_split) & (y.index<=test_end_idx)]):
                if i!=0 and (i==-current_trade or current_trade==0):
                    pnl+=j*(-i)
                    num_transactions+=1
                    current_trade=i
                    #print(i,j,pnl)
                close=j
            if num_transactions%2==1:
                pnl+=close*current_trade
                num_transactions+=1
'''


# In[36]:


class Strategy2_opt_daytrading(SignalStrategy):

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
        if self.data.y_pred[-1]==1:
            # if self.position.is_short:
            #    self.position.close()
            #    return
            if not self.position:
#                self.buy(size=math.floor(100000/self.data.Close[-1]), limit=None, stop=None, sl=(1-self.data.sl[-1])*self.data.Close[-1], tp=(1+self.data.tp[-1])*self.data.Close[-1], tag=None)
                self.buy(size=1, limit=None, stop=None, sl=(1-self.data.sl[-1])*self.data.Close[-1], tp=(1+self.data.tp[-1])*self.data.Close[-1], tag=None)

        if self.data.y_pred[-1]==-1:
            # if self.position.is_long:
            #    self.position.close()
            #    return
            if not self.position:
#                self.sell(size=math.floor(100000/self.data.Close[-1]), limit=None, stop=None, tp=(1-self.data.tp[-1])*self.data.Close[-1], sl=(1+self.data.sl[-1])*self.data.Close[-1], tag=None)
                self.sell(size=1, limit=None, stop=None, tp=(1-self.data.tp[-1])*self.data.Close[-1], sl=(1+self.data.sl[-1])*self.data.Close[-1], tag=None)


# In[37]:


def do_backtest4(X_train, X_test, y_train, y_test, data_target, params, weighting):
    rand_int = randint(1000000, 2000000)
    model_xgb = XGBClassifier(num_class=3, device='gpu',
                    learning_rate=params['learning_rate'],
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                    subsample=params['subsample'],
                    gamma=params['gamma'],
                    objective='multi:softprob',
                    random_state=rand_int,
                    # early_stopping_rounds=10,
                    eval_metric='merror',
                    # min_delta=0.01,
                    # num_leaves=params['num_leaves'],
                    # feature_fraction=params['feature_fraction']
                    )                   # model

    sample_weights = compute_sample_weight(
        class_weight='balanced',
        y=y_train
    )
    model_xgb.fit(X_train, y_train, sample_weight=sample_weights)                           # cp.array(X_train)
    y_pred = model_xgb.predict(X_test)

    y_series = pd.Series(y_pred-1, index=X_test.index, name="y_pred")

    data_target = data_target.join(y_series)

#    data_target.to_csv('data_target_optim_1.csv')

    bt = Backtest(data_target,
        Strategy2_opt_daytrading,
        cash=100000,
        spread=0,
        commission=0.0001,
        margin=1,
        trade_on_close=True,
        hedging=False,
        exclusive_orders=True,
        finalize_trades=True)
    stats = bt.run()
    profit = stats['Equity Final [$]']-100000
#    print(stats)
#    scores.append(profit)
#    sharpe.append(stats['Sharpe Ratio'])
#    sortino.append(stats['Sortino Ratio'])
#    calmar.append(stats['Calmar Ratio'])

    return profit, stats['Sharpe Ratio'], stats['Sortino Ratio'], stats['Calmar Ratio'], y_series


# In[38]:


num_splits = 19

mondays_splits = [int(4 + (num_mondays-4) * (random()/2 + i) / num_splits) for i in range(1, num_splits)]
print(mondays_splits)
train_splits = [mondays_indexes[i] for i in mondays_splits]
print(train_splits)

# train_splits = [int(20 + (len(unique_dates)-20) * (random()/2 + i) / num_splits) for i in range(1, num_splits)]
# train_splits


# In[43]:


def objective4(trial):

    # Prepare dataset
    global dax_data
    global splits

    params = {
     'n_estimators': trial.suggest_int('n_estimators', 300, 400),
     'max_depth': trial.suggest_int('max_depth', 7, 8),
     'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.02),
     'subsample': trial.suggest_float('subsample', 0.95, 0.95),
     'gamma':  trial.suggest_float('gamma', 0.95, 0.95),
     # 'feature_fraction':  trial.suggest_float('feature_fraction', 0.9, 1),
     # 'num_leaves':  trial.suggest_int('num_leaves', 10, 200),

     'sma1_period': trial.suggest_int('sma1_period', 2, 20),
     'sma2_period': trial.suggest_int('sma2_period', 30, 90), 
     'bb_periods': trial.suggest_int('bb_periods', 15, 50),
     'bb_nbdev': trial.suggest_float('bb_nbdev', 2, 2.5),
     'ema1_period': trial.suggest_int('ema1_period', 5, 10),
     'ema2_period': trial.suggest_int('ema2_period', 10, 40),
     'sar_acc': trial.suggest_float('sar_acc', 0.01, 0.5), 
     'sar_max': trial.suggest_float('sar_max', 0.1, 1.5), 
     'midprice_window': trial.suggest_int('midprice_window', 2, 2), # 2,30
     'l1_fast': trial.suggest_int('l1_fast', 5, 20), # 15,3,10
     'l2_fast': trial.suggest_int('l2_fast', 2, 6), 
     'l3_fast': trial.suggest_int('l3_fast', 5, 15), 
     'l1_slow': trial.suggest_int('l1_slow', 10, 40), 
     'l2_slow': trial.suggest_int('l2_slow', 2, 6),
     'l3_slow': trial.suggest_int('l3_slow', 15, 25),
     'kama_trend_period': trial.suggest_int('kama_trend_period', 20, 40),

    'ha_candle_period': trial.suggest_int('ha_candle_period', 4, 30), 
    'dc_market_regime_period': trial.suggest_int('dc_market_regime_period', 4, 30), 
    'displacement_strength_period': trial.suggest_int('displacement_strength_period', 2, 40), 
    'displacement_strength': trial.suggest_float('displacement_strength', 1, 2),
    'displacement_hull_period': trial.suggest_int('displacement_hull_period', 10, 60), 
#    'displacement_sma_period': trial.suggest_int('displacement_sma_period', 2, 30), 
    'displacement_hull_slope_period': trial.suggest_int('displacement_hull_slope_period', 5, 20),

    'gap_lookback': trial.suggest_int('gap_lookback', 3, 9),
    'gap_hull_period': trial.suggest_int('gap_hull_period', 4, 40),
    'gap_hull_slope_period': trial.suggest_int('gap_hull_slope_period', 2, 15),

     'market_regime_threshold': trial.suggest_float('market_regime_threshold', 0.0002, 0.003),
     'tenkan_window': trial.suggest_int('tenkan_window', 4, 15), 
     'kijun_window': trial.suggest_int('kijun_window', 10, 80), 
     'cci_timeperiods': trial.suggest_int('cci_timeperiods', 3, 30),
     'macd_fastperiod': trial.suggest_int('macd_fastperiod', 6, 18), 
     'macd_slowperiod': trial.suggest_int('macd_slowperiod', 10, 40), 
     'macd_signalperiod': trial.suggest_int('macd_signalperiod', 3, 15),
     'price_distribution_window_size': trial.suggest_int('price_distribution_window_size', 5, 5),   # 5,50
     'price_distribution_percentile_threshold': trial.suggest_float('price_distribution_percentile_threshold', 0.2, 0.2), # 0.2,0.5
     'rsi_period': trial.suggest_int('rsi_period', 5, 40),
     'rsi_slope_period': trial.suggest_int('rsi_slope_period', 3, 20),
    'stoch_fastk_period': trial.suggest_int('stoch_fastk_period', 2, 15),
    'stoch_slowk_period': trial.suggest_int('stoch_slowk_period', 2, 15),
    'stoch_slowd_period': trial.suggest_int('stoch_slowd_period', 10, 30),
    'ppo_fastperiod': trial.suggest_int('ppo_fastperiod', 3, 15),
    'ppo_slowperiod': trial.suggest_int('ppo_slowperiod', 25, 45),

    'stochrsi_timeperiod': trial.suggest_int('stochrsi_timeperiod', 7, 15),
    'stochrsi_fastk_period': trial.suggest_int('stochrsi_fastk_period', 5, 25),
    'stochrsi_fastd_period': trial.suggest_int('stochrsi_fastd_period', 3, 20),
    'train_range_len': trial.suggest_int('train_range_len', 10, 15),
    'test_range_len': trial.suggest_int('test_range_len', 3, 5),
    'hour_range_start': trial.suggest_int('hour_range_start', 10, 11),
    'hour_range_stop': trial.suggest_int('hour_range_stop', 20, 20),
    'adx_timeperiod': trial.suggest_int('adx_timeperiod', 5, 5),      #5,15
    'di_timeperiod': trial.suggest_int('di_timeperiod', 5, 20),
    'macd_slope_period': trial.suggest_int('macd_slope_period', 9, 9),
    'sl': trial.suggest_float('sl', 0.002, 0.004),
    'tp': trial.suggest_float('tp', 0.002, 0.005),

    'atr_period': trial.suggest_int('atr_period', 3, 3),

    'stochrsik_slope_period': trial.suggest_int('stochrsik_slope_period', 3, 10),
    'stochk_slope_period': trial.suggest_int('stochk_slope_period', 3, 10),
    'willr_timeperiod': trial.suggest_int('willr_timeperiod', 10, 30),

    'ha_sign_ma_period': trial.suggest_int('ha_sign_ma_period', 4, 15),

    'target_tp': trial.suggest_float('target_tp', 0.002, 0.005),
    'ema_period': trial.suggest_int('ema_period', 15, 20),
    'ema_reversed_period': trial.suggest_int('ema_reversed_period', 5, 10),
    'threshold_long': trial.suggest_float('threshold_long', 0.8, 0.85),
    'threshold_short': trial.suggest_float('threshold_short', 0.15, 0.2),
    }

    dax_data[index_base].loc[:,"labeling_binary"], dax_data[index_base].loc[:,"labeling_dual_ema"], dax_data[index_base].loc[:,"labeling_multi"] = build_target(dax_data[index_base], \
        close_col='Close_wavelet', high_col='Close_wavelet', low_col='Close_wavelet', high_time_col="high_time", low_time_col="low_time", \
        tp=params['target_tp'], ema_period=params['ema_period'], ema_reversed_period=params['ema_reversed_period'], \
        threshold_long=params['threshold_long'], threshold_short=params['threshold_short'])

    p={}
    p[7] = params
    p[10] = params
    X, y, X_columns = getXy(dax_data, index_base, indexes_higher, params, p, timeframe_scalers, list_X, col_y[0], date(2023,6,1), col_open="Open", col_high="High", col_low="Low", col_close="Close")
    y=y+1

#    X = X.loc[(X.index.hour>=params['hour_range_start']) & (X.index.hour<=params['hour_range_stop'])]
    X.loc[:,'DaytradingExit'] = ((X.index.date != X.index.to_series().shift(periods=-1).dt.date) | (X.index.date != X.index.to_series().shift(periods=-2).dt.date))

    #print(lag_f.get_feature_names_out())

    #ml_data.to_csv('ml_data1.csv')
    num_splits = 19

    # Split the dataset to test and train sets
    # Split the initial 70% of the data as training set and the remaining 30% data as the testing set
    mondays_splits = [int(4 + (num_mondays-4) * (random()/2 + i) / num_splits) for i in range(1, num_splits)]
    train_splits = [mondays_indexes[i] for i in mondays_splits]

    scores = []
    sharpe = []
    sortino = []
    calmar = []
    splits_all.append(train_splits)
    total_score = 0
    print(datetime.now().strftime('%H:%M:%S'))

    with mp.Pool(6) as p:
        tuples = []
        for idx,i in enumerate(train_splits):

            train_split = unique_dates[i]
            train_start_idx = unique_dates[max(i-params['train_range_len']*5, 0)]
            test_end_idx = unique_dates[min(i+params['test_range_len']*5, len(unique_dates)-10)]

            X_train, X_test = X.loc[(X.index.date>train_start_idx) & (X.index.date<=train_split)], X.loc[(X.index.date>train_split) & (X.index.date<=test_end_idx)]
            y_train, y_test = y[(X.index.date>train_start_idx) & (X.index.date<=train_split)], y[(X.index.date>train_split) & (X.index.date<=test_end_idx)]

            data_target = X_test.loc[:,['Open','High','Low','Close','DaytradingExit']]
            data_target['sl']=params['sl']
            data_target['tp']=params['tp']

#            data_target = data_target.join(ml_data['atr']).ffill().bfill()
            tuples.append((X_train, X_test, y_train, y_test, data_target, params, exponential_growth(1, 0.02, num_splits-idx-1)))

        results = p.starmap(do_backtest4, tuples)

#    y_pred_all = pd.Series(name='y_pred', index=X.index)

    for res in results:
        scores.append(res[0])
        sharpe.append(res[1])
        sortino.append(res[2])
        calmar.append(res[3])
#        y_pred_all = y_pred_all.combine_first(res[4])
#    y_pred_all.to_csv('y_pred_all_opt1.csv')

    total_score = sum(scores)
    scores_std=np.std(scores)
    print(f'Splits: {train_splits}')
    print(f'Profits: {scores} Sum: {total_score} Stddev: {scores_std}')

    print(f'Sharpe: {sharpe} Avg: {np.mean(sharpe)}')
    print(f'Sortino: {sortino} Avg: {np.mean(sortino)}')
    print(f'Calmar: {calmar} Avg: {np.mean(calmar)}')

    return total_score + np.mean(np.sort(scores)[:3]) #* np.sqrt(max(np.mean(sortino)+np.mean(calmar), 1)) / (parameters['hour_range_stop']-parameters['hour_range_start']+2)


# In[44]:


splits_all = []
study = optuna.create_study(direction='maximize')
study.optimize(objective4, n_trials=10)


# In[45]:


print(study.best_trial.number)
print(study.best_params)
print("Best score:", study.best_value)

parameters_all = study.best_params
splits_best_trial = splits_all[study.best_trial.number]
print(splits_best_trial)


# In[46]:


trials = study.trials_dataframe()
trials.sort_values(by=['value'], ascending=False)
#np.mean(np.sort(trials['value'])[:3])


# In[47]:


plot_optimization_history(study)


# In[48]:


plot_parallel_coordinate(study)


# In[ ]:


get_param_importances(study)


# In[ ]:


plot_parallel_coordinate(study, params=[
 "bb_periods", "bb_nbdev",
'tenkan_window', 'kijun_window',
    'cci_timeperiods', 
    'macd_fastperiod', 'macd_slowperiod', 'macd_signalperiod',
    'rsi_period',
    'stoch_fastk_period', 'stoch_slowk_period', 'stoch_slowd_period',
    'ppo_fastperiod', 'ppo_slowperiod',
    'stochrsi_timeperiod', 'stochrsi_fastk_period', 'stochrsi_fastd_period',
                                       ])


# In[50]:


plot_contour(study, params=[
 'train_range_len',#'test_range_len',
 'hour_range_stop','hour_range_start'
                                       ])


# In[ ]:


plot_contour(study, params=['sl','tp'])


# In[ ]:


plot_contour(study, params=['train_range_len','test_range_len'])


# In[ ]:


plot_contour(study,  params=["bb_periods", "bb_nbdev", ])


# In[ ]:


plot_contour(study,  params=['macd_fastperiod', 'macd_slowperiod', 'macd_signalperiod', ])


# In[ ]:


plot_slice(study,  params=["bb_periods", "cci_timeperiods"])


# In[49]:


plot_param_importances(study)


# In[ ]:


study


# In[50]:


# Take the data
dfc = dax_data[index_base].loc[ (dax_data[index_base].index.date>=unique_dates[380]) & (dax_data[index_base].index.date<unique_dates[385]), :]

fig, ax1 = plt.subplots(figsize=(12, 8))

color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('price', color=color)
ax1.plot(dfc.index, dfc['Close'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('target', color=color)  # we already handled the x-label with ax1
ax2.plot(dfc.index, dfc[["labeling_dual_ema"]], color=color)
ax2.plot(dfc.index, dfc[["labeling_multi"]], color='green')
ax2.tick_params(axis='y', labelcolor=color)
plt.axhline(parameters['threshold_long']*2-1,color='g')
plt.axhline(parameters['threshold_short']*2-1,color='r')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# # Simulation

# In[62]:


def simulation(params, splits):

    dax_data[index_base].loc[:,"labeling_binary"], dax_data[index_base].loc[:,"labeling_dual_ema"], dax_data[index_base].loc[:,"labeling_multi"] = build_target(dax_data[index_base], \
        close_col='Close_wavelet', high_col='Close_wavelet', low_col='Close_wavelet', high_time_col="high_time", low_time_col="low_time", \
        tp=params['target_tp'], ema_period=params['ema_period'], ema_reversed_period=params['ema_reversed_period'], \
        threshold_long=params['threshold_long'], threshold_short=params['threshold_short'])

    p={}
    p[7] = params
    p[10] = params
    X, y, X_columns = getXy(dax_data, index_base, indexes_higher, params, p, timeframe_scalers, list_X, col_y[0], date(2023,6,1), col_open="Open", col_high="High", col_low="Low", col_close="Close")
    y=y+1
    X.loc[:,'DaytradingExit'] = ((X.index.date != X.index.to_series().shift(periods=-1).dt.date) | (X.index.date != X.index.to_series().shift(periods=-2).dt.date)).fillna(True)
    X.to_csv('X_all_sim1.csv')
    print(X.columns)
    model_xgb = XGBClassifier(num_class=3, device='gpu',
                        learning_rate=params['learning_rate'],
                        n_estimators=params['n_estimators'],
                        max_depth=params['max_depth'],
                        subsample=params['subsample'],
                        gamma=params['gamma'],
                        objective='multi:softprob',
                        random_state=100)                   # model

    num_splits = 9
    if len(splits)>0:
        train_splits = splits
        num_splits = len(splits)+1
    else:
        mondays_splits = [int(4 + (num_mondays-4) * (random()/2 + i) / num_splits) for i in range(1, num_splits)]
        train_splits = [mondays_indexes[i] for i in mondays_splits]

    print(datetime.now().strftime('%H:%M:%S'))
    print(train_splits)

    y_pred_all = pd.Series(name='y_pred', index=X.index)

    for i in train_splits:
        train_split=unique_dates[i]
        train_start_idx = unique_dates[ max(i-params['train_range_len']*5, 0) ]
        test_end_idx = unique_dates[min(i+params['test_range_len']*5, len(unique_dates)-10)]

        X_train, X_test = X.loc[(X.index.date>train_start_idx) & (X.index.date<=train_split)], X.loc[(X.index.date>train_split) & (X.index.date<=test_end_idx)]
        y_train, y_test = y[(X.index.date>train_start_idx) & (X.index.date<=train_split)], y[(X.index.date>train_split) & (X.index.date<=test_end_idx)]

        model_xgb.fit(X_train, y_train)
        y_pred = model_xgb.predict(X_test)

        y_series = pd.Series(y_pred-1, index=X_test.index, name="y_pred")

#        print((y_test-1).values)
#        print(y_series.values)
#        print(((y_test-1).values*y_series.values))
        print(((y_test-1).values*y_series.values).sum())

        y_pred_all = y_pred_all.combine_first(y_series)

    y_pred_all.to_csv('y_pred_all_sim1.csv')

    y_pred_all = y_pred_all.loc[ (y_pred_all.index.date>unique_dates[train_splits[0]]) & (y_pred_all.index.date<=unique_dates[min(train_splits[-1]+params['test_range_len'], len(unique_dates)-1)]) ]
#    print(y_pred_all)
#    print(y_pred_all.sum())

#    data_target = dax_data_10s.loc[(dax_data_10s.index.hour>=params['hour_range_start']) & (dax_data_10s.index.hour<=params['hour_range_stop'])]
    #data_target = data_target.loc[ (data_target.index>pd.to_datetime( unique_dates[ max(train_splits[1]-parameters['train_range_len'], 0) ] )) & (data_target.index<=pd.to_datetime(unique_dates[min(train_splits[num_splits-1]+parameters['test_range_len'], len(unique_dates)-1)])) ]    

#    data_target = data_target.loc[(data_target.index>=y_pred_all.index[0]) & (data_target.index<=y_pred_all.index[-1]), :]
#    data_target.loc[:,'DaytradingExit'] = ((data_target.index.date != data_target.index.to_series().shift(periods=-1).dt.date) | (data_target.index.date != data_target.index.to_series().shift(periods=-2).dt.date)).fillna(True)
    data_target = X.loc[:,['Open','High','Low','Close']]
    data_target['DaytradingExit'] = X['DaytradingExit']

#    data_target['hour_range_stop']=parameters['hour_range_stop']
    data_target.loc[:,'sl']=params['sl']
    data_target.loc[:,'tp']=params['tp']
    data_target = data_target.join(y_pred_all.rename('y_pred'))
    #data_target = data_target.join(ml_data['atr']).ffill().bfill()

    data_target.to_csv('data_target_sim.csv')
    bt = Backtest(data_target,
        Strategy2_opt_daytrading,
        cash=100000,
        spread=0,
        commission=0.0001,
        margin=1,
        trade_on_close=False,
        hedging=False,
        exclusive_orders=True,
        finalize_trades=True)
    return bt


# In[63]:


bt = simulation(parameters_all, splits_best_trial)    # splits_best_trial
stats = bt.run()
print(stats)


# In[70]:


bt.plot()


# In[71]:


#equity_splits = [stats.loc[unique_dates[i], '_equity_curve'] for i in splits_best_trial]
#[unique_dates[i] for i in splits_best_trial]
#stats.loc[unique_dates[1], '_equity_curve']


# In[72]:


stats['_equity_curve']


# In[73]:


equity = (stats['_equity_curve'].groupby(stats['_equity_curve'].index.ceil('5min'))
            .agg(
                Equity=('Equity','last'),
                Drawdown=('DrawdownPct','last'),
                DrawdownDuration=('DrawdownDuration','last')))


# In[74]:


equity['Equity'].plot()


# In[75]:


stats['_trades']


# In[ ]:





# In[ ]:


data_target = dax_data_10s.loc[(dax_data_10s.index.dayofweek<=4)]
data_target = data_target.loc[(data_target.index.hour>=parameters['hour_range_start']) & (data_target.index.hour<=parameters['hour_range_stop'])]


# In[ ]:


data_target['DaytradingExit'] = (data_target.index.to_series().dt.date != data_target.index.to_series().shift(periods=-1).dt.date)


# In[ ]:


def supertrend(dataframe: pd.DataFrame, multiplier, atr_period=14, close_col="close", high_col="high", low_col="low"):

#    h_l = dataframe[high_col] - dataframe[low_col]
#    h_pc = np.absolute(dataframe[high_col] - dataframe[close_col].shift(1))
#    l_pc = np.absolute(dataframe[low_col] - dataframe[close_col].shift(1))

    tr = talib.TRANGE(dataframe[high_col], dataframe[low_col], dataframe[close_col])
#    tr = np.maximum(h_l, np.maximum(h_pc, l_pc))
    atr = talib.SMA(tr, atr_period) * multiplier

    close = dataframe[close_col].to_numpy()

    # Compute basic upper and lower bands
    avg = (dataframe[high_col] + dataframe[low_col]) / 2
    basic_ub = (avg + atr).to_numpy()
    basic_lb = (avg - atr).to_numpy()

    # Compute final upper and lower bands
    n = len(dataframe)
    final_ub = np.full(n, 0, dtype=np.float32)
    final_lb = np.full(n, 0, dtype=np.float32)

    for i in range(atr_period, n):
        final_ub[i] = basic_ub[i] if basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1] else final_ub[i-1]
    for i in range(atr_period, n):
        final_lb[i] = basic_lb[i] if basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1] else final_lb[i-1]

    # Set the Supertrend value
    st = np.full(n, 0, dtype=np.float32)

    for i in range(atr_period, n):
        st[i] = (
            final_ub[i] if st[i-1] == final_ub[i-1] and close[i] <= final_ub[i] else
            final_lb[i] if st[i-1] == final_ub[i-1] and close[i] > final_ub[i] else
            final_lb[i] if st[i-1] == final_lb[i-1] and close[i] >= final_lb[i] else
            final_ub[i] if st[i-1] == final_lb[i-1] and close[i] < final_lb[i] else
            0
        )
    # Mark the trend direction up/down
    stx = np.where(close < st, -1, 1)
    stx_strength = stx* np.absolute(st-close)/atr
    return st, stx, stx_strength


# In[ ]:


dax_data[index_base].loc[:, f"st_1"], dax_data[index_base].loc[:, f"stx_1"], dax_data[index_base].loc[:, f"stx_strength_1"] = supertrend(
            dax_data[index_base], 10, 40, close_col="Close", high_col="High", low_col="High"
        )


# In[ ]:


plt.figure(figsize=(14, 10))
plt.plot(dax_data[index_base].loc[:,['st_1', 'Close']].iloc[22000:22500,])

plt.show()


# In[ ]:


plt.figure(figsize=(14, 10))
plt.plot(dax_data[index_base].loc[:,['stx_1', 'stx_strength_1']].iloc[22000:22500,])

plt.show()


# ## Tests

# In[ ]:


x = dax_data[index_base].iloc[ (dax_data[index_base].index.date>=unique_dates[378]) & (dax_data[index_base].index.date<unique_dates[382]), :]
x1 = dax_data[index_barrier].iloc[ (dax_data[index_barrier].index.date>=unique_dates[378]) & (dax_data[index_barrier].index.date<unique_dates[380]), :]
print(len(x1))

# Calculate the Hurst exponent
hurst, ci, data = compute_Hc(x['Close'])
print("Hurst exponent:",hurst)


# In[ ]:


### Function to calculate the Hurst exponent of a time series
def hurst_exponent(series):
    try:
        H, c, data = compute_Hc(series, kind='price', simplified=True)
    except:
        H = np.nan
    return H

def hurst_exponent_change(series):
    try:
        H, c, data = compute_Hc(series, kind='change', simplified=True)
    except:
        H = np.nan
    return H

def hurst_calc(df: pd.DataFrame, col: str, window_size: int = 100) -> pd.DataFrame:
    return df.loc[:,col].rolling(window=window_size, min_periods=window_size).apply(hurst_exponent, raw=False)
def hurst_calc_change(df: pd.DataFrame, col: str, window_size: int = 100) -> pd.DataFrame:
    return df.loc[:,col].rolling(window=window_size, min_periods=window_size).apply(hurst_exponent_change, raw=False)


# In[ ]:


df = pd.DataFrame(index=x.index)
df1 = pd.DataFrame(index=x1.index)
df["hurst"] = fe.math.hurst(df=x, col="Close", window_size=100)
df['hurst2'] = hurst_calc(df=x, col="Close", window_size=100)
df1['hurst3'] = hurst_calc_change(df=x1, col="log_ret_ha_1", window_size=100)


# In[ ]:


fig, ax1 = plt.subplots(figsize=(12, 8))

color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('price', color=color)
ax1.plot(x.index, x['Close'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('target', color=color)  # we already handled the x-label with ax1
ax2.plot(df.index, df[["hurst"]], color=color)
ax2.plot(df1.index, df1[["hurst3"]], color='green')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# In[ ]:


fig, ax1 = plt.subplots(figsize=(12, 8))

color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('price', color=color)
ax1.plot(x1.index, x1['ha_close'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('target', color=color)  # we already handled the x-label with ax1
ax2.plot(df1.index, df1[["hurst3"]], color=color)
#ax2.plot(df1.index, df1[["hurst4"]], color='green')
ax2.tick_params(axis='y', labelcolor=color)
plt.axhline(0.6,color='g')
plt.axhline(0.35,color='r')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# In[ ]:


# Kalman filter
from pykalman import KalmanFilter
kf = KalmanFilter(transition_matrices = [1],
                  observation_matrices = [1],
                  initial_state_mean = x['ha_close'].iloc[0],
                  initial_state_covariance = 1,
                  observation_covariance=1,
                  transition_covariance=.01)
df['ha_close_kalman'], _ = kf.filter(x['ha_close'])


# In[ ]:


kf = KalmanFilter(transition_matrices = [1],
                  observation_matrices = [1],
                  initial_state_mean = x['Close'].iloc[0],
                  initial_state_covariance = 1,
                  observation_covariance=1,
                  transition_covariance=.01)
df['close_kalman'], _ = kf.filter(x['Close'])


# In[ ]:


kf = KalmanFilter(transition_matrices = [1],
                  observation_matrices = [1],
                  initial_state_mean = 0.5,
                  initial_state_covariance = 1,
                  observation_covariance=1,
                  transition_covariance=.01)
df['hurst_kalman'], _ = kf.filter(df['hurst'].fillna(0.5))


# In[ ]:


#df.loc[:,'ha_close_kalman2'] = x['ha_close'].rolling(window=100).apply(kf.filter)


# In[ ]:


#df['state_means_hurst'] #.isnull().sum()


# In[ ]:


fig, ax1 = plt.subplots(figsize=(12, 8))

color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('price', color=color)
ax1.plot(x.index, x['Close'], color=color)
ax1.plot(x.index, df['ha_close_kalman'], color='green')
ax1.plot(x.index, df['close_kalman'], color='blue')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

ax2.set_ylabel('target', color=color)  # we already handled the x-label with ax1
#ax2.plot(x.index, x[["kama_regime_slow"]], color='blue')
ax2.plot(df.index, df[["hurst_kalman"]], color='black')
#ax2.plot(df.index, df[["state_means_hurst"]], color='green')
#ax2.plot(df.index, df[["hurst"]], color='pink')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# In[ ]:


import pywt
pywt.families(short=False)


# In[ ]:


# Perform the Discrete Wavelet Transform with 'dbN'


# In[ ]:


wavelet_coeff = wavelet_transform(x['ha_close'], 8)
reconstructed_data = inverse_wavelet_transform(wavelet_coeff, 8, 4)
reconstructed_data2 = wavelet_denoising2(x['Close'], wavelet='db6', lvl=8, clear_levels=3)


# In[ ]:


reconstructed_data3 = x['Close'].rolling(window=10, min_periods=10).apply(wavelet_denoising3, raw=True, args=('db6', 8, 7, None)).fillna(x['Close'].iloc[0])
reconstructed_data4 = x['ha_close'].rolling(window=10, min_periods=10).apply(wavelet_denoising3, raw=True, args=('db6', 7, 5, None)).fillna(x['ha_close'].iloc[0])


# In[ ]:


fig, ax1 = plt.subplots(figsize=(12, 8))

color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('price', color=color)
ax1.plot(x.iloc[50:200].index, x.iloc[50:200]['Close'], color=color)

#ax1.plot(x.index, x[["kama_regime_fast"]], color='blue')
#ax1.plot(x.iloc[50:150].index, reconstructed_data[50:150], color='green')
ax1.plot(x.iloc[50:150].index, reconstructed_data2[50:150], color='yellow')
ax1.plot(x.iloc[50:150].index, reconstructed_data3[50:150], color='black')
ax1.plot(x.iloc[50:150].index, reconstructed_data4[50:150], color='green')

ax1.plot(x.iloc[50:150].index, df['close_kalman'][50:150], color='orange')

ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

ax2.set_ylabel('target', color=color)  # we already handled the x-label with ax1
# ax2.plot(x.index, x[["kama_regime_fast"]], color='blue')
#ax2.plot(x.index, (reconstructed_data[:-1]/x['Close'])**1, color='black')
ax2.plot(x.iloc[50:150].index, x.iloc[50:150][["kama_trend_slow_diff"]], color='teal')
ax2.plot(x.iloc[50:150].index, x.iloc[50:150][["kama_trend_slow_diff"]]+x.iloc[50:150][["kama_trend_slow_diff"]].diff(), color='pink')
plt.axhline(1.002,color='g')
plt.axhline(0.998,color='r')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# In[ ]:


fig, ax1 = plt.subplots(figsize=(12, 8))

color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('price', color=color)
ax1.plot(x.index, x['Close'], color=color)

ax1.plot(x.index, x["kama_regime_fast"], color='blue')
#ax1.plot(x.index, reconstructed_data[:-1], color='green')

ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

ax2.set_ylabel('target', color=color)  # we already handled the x-label with ax1
# ax2.plot(x.index, x[["kama_regime_fast"]], color='blue')
ax2.plot(x.index, x["kama_regime_fast"]/x['ha_close'], color='black')
#ax2.plot(df.index, df[["hurst"]], color='green')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# In[ ]:


from statsmodels.tsa.stattools import acf, pacf, q_stat, adfuller, grangercausalitytests

# Augmented Dickey–Fuller test for stationarity
result = adfuller(x['Close'],regression ='ctt')
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
if result[1]  > 0.05 :
    print('Fail to reject the null hypothesis (H0), the data has a unit root and is non-stationary.')
elif result[1] <= 0.05 :
    print('Reject the null hypothesis (H0), the data does not have a unit root and is stationary.')


# In[ ]:




