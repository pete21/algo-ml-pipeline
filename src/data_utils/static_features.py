from datetime import datetime

import numpy as np
import pandas as pd
import talib

from src.data_utils.features import (
    candle_information,
    cycle,
    diff_transform,
    heikenashi_open,
    hurst_calc_change,
    volatility,
)
from src.data_utils.features_engineering import math
from src.data_utils.wavelet import wavelet_denoising_rolling


def sliding_elementwise_mult(values: np.ndarray, weights: np.ndarray, scaler: float) -> float:
    """Elementwise multiply window values by weights; return sum of products."""
    return float(np.dot(values, weights)/np.mean(values)*scaler)

def static_features(df: pd.DataFrame, scaler: float, high_col: str="high", low_col: str="low", open_col: str="open", close_col: str="close") -> pd.DataFrame:
    print(datetime.now().strftime('%H:%M:%S'))

    # df.loc[:,'hour_sin'] = np.sin((df.index.hour * 60 + df.index.minute) * np.pi / 720)
    # df.loc[:,'hour_cos'] = np.cos((df.index.hour * 60 + df.index.minute) * np.pi / 720)
    # df.loc[:,'dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    # df.loc[:,'dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    df.loc[:,'minute_of_day'] = df['local_date'].dt.hour * 60 + df['local_date'].dt.minute

    df.loc[:,'hour_sin'] = np.sin(df['minute_of_day'] * np.pi / 720)
    df.loc[:,'hour_cos'] = np.cos(df['minute_of_day'] * np.pi / 720)
    df.loc[:,'dow_sin'] = np.sin(2 * np.pi * df['local_date'].dt.dayofweek / 7)
    df.loc[:,'dow_cos'] = np.cos(2 * np.pi * df['local_date'].dt.dayofweek / 7)

    # df.loc[:,'day_of_week'] = df.index.dayofweek / 2 - 1

    # Heiken-ashi
    df.loc[:,'ha_close'] = (df[open_col] + df[high_col] + df[low_col] + df[close_col]) / 4
    df.loc[:,'ha_open'] = heikenashi_open(df['ha_close'].to_numpy(), ha_open_0=(df[open_col].iloc[0] + df[close_col].iloc[0]) / 2)
    df.loc[:,'ha_high'] = np.maximum(df[high_col], df['ha_open'])
    df.loc[:,'ha_low'] = np.minimum(df[low_col], df['ha_open'])
    df.loc[:,'ha_mid'] = (df['ha_open']+df['ha_close'])/2

    df.loc[:,'log_close'] = np.log(df[close_col])
    df.loc[:,'log_ha_close'] = np.log(df['ha_close'])

    df.loc[:,'sine'], df.loc[:,'leadsine'], df.loc[:,'integer'] = cycle(df, 'ha_close')
    df.loc[:,'sine_diff'] = df['sine']-df['leadsine']

    for i in [2,5,10]:
        df.loc[:,f'ha_slope_{i}'] = talib.LINEARREG_ANGLE(df['log_ha_close'], i)*10/scaler
        df.loc[:,f'sine_slope_{i}'] = talib.LINEARREG_ANGLE(df['sine'], i)/10
        df.loc[:,f'sine_diff_slope_{i}'] = talib.LINEARREG_ANGLE(df['sine_diff'], i)/10


    autocorrelations=[(2,4), (4,8), (6,12)] #, (12,24)]
    for i,b in autocorrelations:
        df.loc[:,f'corr_{i}'] = talib.CORREL(df["ha_close"], df["ha_close"].shift(i), b)             #col_close

    for i in [1,2,3]:
        df.loc[:,f"log_ret_{i}"] = diff_transform(df, 'log_close', i)*10/scaler
        df.loc[:,f"log_ret_ha_{i}"] = diff_transform(df, 'log_ha_close', i)*10/scaler

#    df['velocity'], df['acceleration'] = derivatives(df, "log_ha_close")

    df.loc[:,'candle_sign'], df.loc[:,'candle_filling'], df.loc[:,'body_amplitude'], df.loc[:,'ha_wickstrength'], df.loc[:,'ha_sign'], df.loc[:,'ha_candle_fill'] = candle_information(df)

    for i in [2, 4, 8, 12]:
        df.loc[:,f'rogers_satchell_vol_{i}'], df.loc[:,f'parkinson_vol_{i}'], df.loc[:,f'yang_zhang_vol_{i}'], df.loc[:,f'ctc_vol_{i}'] \
            = volatility(df, i, i, i, i, high_col, low_col, open_col, close_col)

    #vol_features = [col for col in dax_data[index_base].columns if "vol_" in col]
    short_term_vol_features = ['rogers_satchell_vol_2','rogers_satchell_vol_4'] #,'yang_zhang_vol_3','yang_zhang_vol_6',]
    long_term_vol_features = ['rogers_satchell_vol_8', 'rogers_satchell_vol_12'] #,'yang_zhang_vol_20']
    #print(unique_weekdates)
    #print(short_term_vol_features)
    #print(long_term_vol_features)

    #PCA
    # print("PCA short start: "+datetime.now().strftime('%H:%M:%S'))
    # df.loc[:,'vol_short_pca1'] = calc_kernel_pca(df, unique_weekdates[1:], 10, short_term_vol_features, ['vol_short_pca1'])
    # df.loc[:,'vol_short_pca1'] = df['vol_short_pca1'].fillna(0)
    # print("PCA short end: "+datetime.now().strftime('%H:%M:%S'))

    # print("PCA long start: "+datetime.now().strftime('%H:%M:%S'))
    # df.loc[:,'vol_long_pca1'] = calc_kernel_pca(df, unique_weekdates[1:], 10, long_term_vol_features, ['vol_long_pca1'])
    # df.loc[:,'vol_long_pca1'] = df['vol_long_pca1'].fillna(0)
    # print("PCA long end: "+datetime.now().strftime('%H:%M:%S'))

    # wavelet_coeff = wavelet_transform(df['ha_close'], 8)
    # df.loc[:,'wavelet_reconstr'] = inverse_wavelet_transform(wavelet_coeff, 8, 4)

    print("hurst: "+datetime.now().strftime('%H:%M:%S'))
    df.loc[:,'hurst'] = hurst_calc_change(df=df, col="log_ret_ha_1", window_size=100)
    # df.loc[:,'hurst_kalman'] = kalman_filter(df['hurst'], 0.5)

    # print("wavelet: "+datetime.now().strftime('%H:%M:%S'))
    df.loc[:,'close_wavelet_rolling'] = df['Close'].rolling(window=10, min_periods=10).apply(wavelet_denoising_rolling, raw=True, args=('db6', 8, 7, None))

    df.loc[:,"abs_log_ret_1"] = np.abs(df["log_ret_1"])
    df.loc[:,"tail_index_1"] = np.log(math.tail_index(df=df, col="abs_log_ret_1", window_size=24, k_ratio=0.10).replace([np.inf], np.nan).ffill())

    df.loc[:,'close_regr_entropy'] = math.sample_entropy(df=df, col='Close', window_size=48)-1
    df.loc[:,'permutation_entropy'] = math.permutation_entropy(df=df, col="Close", window_size=48, order=5)-0.5
    df.loc[:,"skew"] = math.skewness(df=df, col="log_ret_1", window_size=48)
    df.loc[:,"petrosian_fd"] = (math.petrosian_fd(df=df, col="Close", window_size=48)-1)*10

    # Pivots
    print("pivots: "+datetime.now().strftime('%H:%M:%S'))
    long_pivot = np.array([1,0,-1,0,1])-np.mean([1,0,-1,0,1])
    short_pivot = np.array([-1,0,1,0,-1])-np.mean([-1,0,1,0,-1])
    df.loc[:,'long_pivot'] = df.loc[:,"ha_low"].rolling(window=5, min_periods=5).apply(sliding_elementwise_mult, raw=True, kwargs={"weights": long_pivot, "scaler": 1000/scaler})
    df.loc[:,'short_pivot'] = df.loc[:,"ha_high"].rolling(window=5, min_periods=5).apply(sliding_elementwise_mult, raw=True, kwargs={"weights": short_pivot, "scaler": 1000/scaler})

    print(datetime.now().strftime('%H:%M:%S'))
    return df
