import talib
from hurst import compute_Hc
from scipy.signal import hilbert
from math import exp
import numpy as np
# import polars as pl
import pandas as pd
from pandas import DataFrame, Series
#from functools import lru_cache
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
import numba
from src.data_utils.wavelet import wavelet_denoising_rolling
from pykalman import KalmanFilter
from src.data_utils.features_engineering.volatility.close_to_close import close_to_close_volatility
from src.data_utils.target_engineering.directional.barriers import double_barrier_labeling, triple_barrier_labeling
from src.data_utils.features_engineering.volatility.range_estimators import rogers_satchell_volatility
from src.data_utils.features_engineering.volatility.range_estimators import parkinson_volatility
from src.data_utils.features_engineering.volatility.range_estimators import yang_zhang_volatility
import src.data_utils.features_engineering.trend as trend
import src.data_utils.features_engineering.math as math


### Target

def build_target(df, open_col="open", high_col="high", low_col="low", high_time_col="high_time",
    low_time_col="low_time", tp=0.0025, ema_period=10, ema_reversed_period=40, threshold_long=0.75, threshold_short=0.25):
    
    # labeling_binary = double_barrier_labeling(df, open_col, high_col, low_col, high_time_col, low_time_col, tp=tp, sl=-tp, buy=True)
    labeling_binary = triple_barrier_labeling(df, 4, open_col, high_col, low_col, high_time_col, low_time_col, tp=tp, sl=-tp, buy=True)

    labeling_ema = talib.EMA(labeling_binary, ema_period)

    labeling_reversed = labeling_binary[::-1]
    labeling_ema_reversed = talib.EMA(labeling_reversed, ema_reversed_period)
    labeling_dual_ema = np.roll((labeling_ema + labeling_ema_reversed)/2, 3)
    # labeling_dual_ema.fillna(0, inplace=True)
    labeling_multi = np.where(labeling_dual_ema>=threshold_long*2-1, 1, np.where(labeling_dual_ema<=threshold_short*2-1, -1, 0))

    return labeling_binary, labeling_dual_ema, labeling_multi


### Candle

# def close_price_distribution(
#         df: pd.DataFrame,
#         col: str,
#         window_size: int = 60,
#         start_percentage: float = 0.25,
#         end_percentage: float = 0.75,
# ) -> pd.Series:
#     return (
#         df[col]
#         .rolling(window_size)
#         .apply(lambda x: _close_percentage_in_range(x, start_percentage, end_percentage), raw=True)
#     )


def candle_information(df):
    # Candle color
    candle_sign = np.sign(df['Close'] - df['Open']) #, dtype=np.float16, name='candle_sign')

    # Filling percentage
    candle_filling = np.where(df['High'] != df['Low'], (df['Close'] - df['Open']) / (df['High'] - df['Low']), 0.5) #, dtype=np.float16, name='candle_filling')

    # Amplitude
    body_amplitude = np.abs(df['Close'] - df['Open']) #, dtype=np.float16, name='body_amplitude')

    # Wick percentage
    maxOpenClose = np.maximum(df['ha_close'], df['ha_open'])
    minOpenClose = np.minimum(df['ha_close'], df['ha_open'])
    ha_topwick = (df['ha_high']-maxOpenClose)/maxOpenClose*1000            # for 5 min bars average wicks: =1 - strength, =0 - no wick
    ha_bottomwick = (minOpenClose-df['ha_low'])/minOpenClose*1000            # for 5 min bars average wicks: =1 - strength, =0 - no wick
    ha_wickstrength = np.sqrt(ha_topwick)-np.sqrt(ha_bottomwick) #, index=df.index, dtype=np.float16, name='ha_wickstrength')
    #wick_strength = ((minOpenClose-df['ha_low'])/(df['ha_high']-maxOpenClose))

    ha_sign = np.sign(df['ha_close']-df['ha_open'])


    ha_range = df['ha_high'] - df['ha_low']
    ha_candle_fill = np.array(np.where(ha_range > 0, abs(df['ha_close']-df['ha_open']) / ha_range, 0.5), dtype='float')
#    ema_ha_upper_wick = np.array(np.where(ha_range > 0, (df['ha_high'] - np.maximum(df['ha_open'], df['ha_close'])) / ha_range, 0.5), dtype='double')
#    ema_ha_lower_wick = np.array(np.where(ha_range > 0, (np.minimum(df['ha_open'], df['ha_close']) - df['ha_low']) / ha_range, 0.5), dtype='double')

    return candle_sign, candle_filling, body_amplitude, ha_wickstrength, ha_sign, ha_candle_fill


'''
def internal_bar_strength(
    df: pd.DataFrame, high_col: str = "high", low_col: str = 'Low', close_col: str = "close"
) -> pd.Series:
    Compute the Internal Bar Strength (IBS) indicator.

    The IBS is defined as:
        IBS = (Close - Low) / (High - Low)

    It measures where the closing price is located within the day's range,
    and is commonly used to detect short-term overbought or oversold conditions.
'''


### Math
def derivatives(df, col):
    """
    Calculates the first and second derivatives of a given column in a DataFrame
    and adds them as new columns 'velocity' and 'acceleration'.

    Parameters:
    -----------
    df : polars.DataFrame
        The DataFrame containing the column for which derivatives are to be calculated.

    col : str
        The column name for which the first and second derivatives are to be calculated.

    Returns:
    --------
    df : polars.DataFrame
        A new DataFrame with 'velocity' and 'acceleration' columns added.

    """

    velocity = df[col].diff().fillna(0).rename('velocity') * 1000
    acceleration = velocity.diff().fillna(0).rename('acceleration')

    return velocity, acceleration


def diff_transform(df: pd.DataFrame, col: str, window_size: int) -> pd.Series:
    """
    Compute the logarithmic percentage change (log return) over a specified window.

    Mathematically:
        log_pct = log(P_t) - log(P_{t - window_size})
                   = log(P_t / P_{t - window_size})

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the column to be analyzed.
    col : str
        The name of the column containing price or value data.
    window_size : int
        The number of periods over which to compute the log return.

    Returns
    -------
    pd.Series
        A Series containing the rolling log returns over `window_size` periods.
    """

    return df[col].diff(window_size) * 100


def auto_corr(df, col, n=40, lag=10):
    """
    Calculates the autocorrelation for a given column in a polars DataFrame, using a specified window size and lag.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing the column for which to compute autocorrelation.
    - col (str): The name of the column in the DataFrame for which to calculate autocorrelation.
    - n (int, optional): The size of the rolling window for calculation. Default is 50.
    - lag (int, optional): The lag step to be used when computing autocorrelation. Default is 10.

    Returns:
    - pd.DataFrame: A new DataFrame with an additional column named 'autocorr_{lag}', where {lag} is the provided lag value. This column contains the autocorrelation values.
    """
    return df[col].rolling(window=n, min_periods=n, center=False).apply(lambda x: x.autocorr(lag=lag), raw=False)


### Statistical

'''
def adf_test(
    df: pd.DataFrame, col: str, window_size: int, lags: int = None, regression: str = "c"
) -> tuple[pd.Series, pd.Series]:

Interpretation

p‑value ≪ 0.05  →  Reject the unit‑root null ⇒ series is stationary in that window.
p‑value ≈ 1  →  cannot reject null ⇒ behaves like a random walk.
Monitor the rolling statistic (adf_stat) to see how strongly the unit‑root hypothesis is rejected (more negative ⇒ stronger evidence of stationarity).

df["adf_stat"], df["adf_pvalue"] = math.adf_test(df, col="close", window_size=80, lags=10, regression="ct")
'''

'''
ARCH Test

df["returns"] = df["close"].pct_change(1)
df["arch_stat"], df["arch_pvalue"] = math.arch_test(df, col="returns", window_size=60, lags=10)
'''

'''
Skewness

df["returns"] = df["close"].pct_change(1)
df["skew"] = math.skewness(df=df, col="returns", window_size=60)
df["skew"]
'''

'''
Kurtosis

df["returns"] = df["close"].pct_change(1)
df["kurt"] = math.kurtosis(df=df, col="returns", window_size=60)
df["kurt"]
'''

# Hurst
#    Around 0.5 --> Random Walk
#    Close to 0.4 or below --> Ranging
#    Close to 0.6 or above --> Trending

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
    # return df.with_columns(
    #     pd.col(col).rolling_apply(hurst_exponent, window_size=window_size, min_periods=window_size)
    # )
def hurst_calc_change(df: pd.DataFrame, col: str, window_size: int = 100) -> pd.DataFrame:
    return df.loc[:,col].rolling(window=window_size, min_periods=window_size).apply(hurst_exponent_change, raw=False)
    # return df.with_columns(
    #     pd.col(col).rolling_apply(hurst_exponent_change, window_size=window_size, min_periods=window_size)
    # )


#df["hurst"] = math.hurst(df=df, col="Close", window_size=200)
#df["hurst"]


'''
# Compute the Features
df["returns"] = df["close"].pct_change(1)
df["vol"] = volatility.parkinson_volatility(df=df, window_size=30)
df["skew"] = math.skewness(df=df, col="returns", window_size=30)

# Compute the Entropy
df["entropy_returns"] = math.sample_entropy(df=df, col="returns", window_size=200)
df["entropy_vol"] = math.sample_entropy(df=df, col="vol", window_size=200)
df["entropy_skew"] = math.sample_entropy(df=df, col="skew", window_size=200)
'''

### Trend

from sklearn.linear_model import LinearRegression

'''
   /* Linear Regression is a concept also known as the
    * "least squares method" or "best fit." Linear
    * Regression attempts to fit a straight line between
    * several data points in such a way that distance
    * between each data point and the line is minimized.
    *
    * For each point, a straight line over the specified
    * previous bar period is determined in terms
    * of y = b + m*x:
    *
    * TA_LINEARREG          : Returns b+m*(period-1)
    * TA_LINEARREG_SLOPE    : Returns 'm'
    * TA_LINEARREG_ANGLE    : Returns 'm' in degree.
    * TA_LINEARREG_INTERCEPT: Returns 'b'
    * TA_TSF                : Returns b+m*(period)
    */
'''
# Function to calculate the slope of the linear regression
def linear_regression_slope(series):
    X = np.arange(len(series)).reshape(-1, 1)  # Create an array of indices for X
    y = series.values.reshape(-1, 1)  # Use the values of the series as y
    model = LinearRegression().fit(X, y)  # Fit the linear regression model
    slope = model.coef_[0][0]  # Extract the slope from the model
    return slope


def trend_regression(df, col='Close', trend_slope_window=10):
    return df[col].rolling(trend_slope_window).apply(linear_regression_slope)


'''
df["returns"] = df["close"].pct_change(1)
df["kurt"] = math.kurtosis(df=df, col="returns", window_size=60)
df["kurt"]
'''

'''
df["kama"] = trend.kama(df=df, col="Close", l1=10, l2=2, l3=30)

df["kama"]
'''

'''
df["linear_slope_1M"] = trend.linear_slope(df, col='close', window_size=30*6) # x6 because we have 4-hour data
df["linear_slope_1M"]
'''


### Cycle
'''
The Hilbert Transform - SineWave indicator, developed by John Ehlers and featured in "Rocket Science for Traders," transforms the dominant cycle phase into two smooth sine wave outputs. The primary Sine output represents the current position within the market cycle, while the LeadSine output is advanced by 45 degrees, providing early signals of impending cycle turns. These waves oscillate between -1 and +1, making cycle tops and bottoms visually obvious.

What sets this indicator apart is its ability to differentiate between cyclical and trending price activity. The sine waves are designed to not cross during trending periods - when you see the waves turn without crossing, it indicates the market has entered a trend. This unique feature combines the best characteristics of oscillators (identifying overbought/oversold in cycles) with moving average properties (signaling trend starts and ends).

Interpretation & Trading Signals

Wave Components:
Sine: Current cycle position as sine wave (-1 to +1)
LeadSine: Sine wave advanced by 45°, crosses 1/8 cycle early
Range: Both oscillate between -1 (cycle trough) and +1 (cycle peak)
Adaptation: Frequency adjusts automatically to dominant cycle

Trading Signals:
Buy Signal: LeadSine crosses above Sine (cycle bottom turning up)
Sell Signal: LeadSine crosses below Sine (cycle top turning down)
Always In Market: Signals switch from long to short and vice versa
Cycle Confirmation: Clear crossovers indicate cycling market

Market Mode Detection:
Cycling Market: Waves cross regularly at extremes
Trending Market: Waves turn without crossing - trend signal!
No Crossover: When sine turns up without crossing bottom = uptrend starting
Hit Rate: Typically achieves ~70% accuracy in cycling markets
'''

def cycle(df, col):
    sine, leadsine = talib.HT_SINE(df[col])
    integer = talib.HT_TRENDMODE(df[col])
    
    # return sine.astype('float16'), leadsine.astype('float16'), integer.astype('int16')
    return sine, leadsine, integer


# Function to calculate Hilbert Transform Dominant Cycle
def hilbert_dominant_cycle(price_series):
    """Compute Hilbert Transform Dominant Cycle."""
    analytic_signal = hilbert(price_series)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    dominant_cycle = np.concatenate(([np.nan], 1 / np.diff(instantaneous_phase)))  # Avoid division by zero
    return pd.Series(dominant_cycle, index=price_series.index).ffill()

    
### Volatility

def volatility(df, rogers_satchell_volatility_window, parkinson_vol_window, yang_zhang_vol_window, ctc_vol_window,
               high_col="High", low_col="Low", open_col="Open", close_col="Close"):
    # Calculate Rogers-Satchell volatility.md estimator using numpy operations with Numba acceleration.
    rogers_satchell_vol = rogers_satchell_volatility(df=df, high_col=high_col,
                                                                                       low_col=low_col,
                                                                                       open_col=open_col,
                                                                                       close_col=close_col,
                                                                                       window_size=rogers_satchell_volatility_window) * 1000-1 #,
                                           #index=df.index, dtype=np.float16, name='rogers_satchell_volatility')

    # Calculate Parkinson's volatility.md estimator using numpy operations with Numba acceleration.
    parkinson_vol = parkinson_volatility(df=df, high_col=high_col, low_col=low_col,
                                                                    window_size=parkinson_vol_window) * 1000-1 #,
                                           #index=df.index, dtype=np.float16, name='rogers_satchell_volatility')

    # Compute Yang-Zhang Volatility over a rolling window
    yang_zhang_vol = yang_zhang_volatility(df=df, window_size=yang_zhang_vol_window,
                                                                      high_col=high_col, low_col=low_col,
                                                                      open_col=open_col, close_col=close_col) * 1000-1 #,
                                           #index=df.index, dtype=np.float16, name='rogers_satchell_volatility')

    # Close-to-close volatility
    ctc_vol = close_to_close_volatility(df=df, close_col=close_col,
                                                                   window_size=ctc_vol_window) * 1000-1 #,
                                           #index=df.index, dtype=np.float16, name='rogers_satchell_volatility')
    
    return rogers_satchell_vol, parkinson_vol, yang_zhang_vol, ctc_vol

### Market regime


def kama_market_regime(df, col="Close", l1_fast=50, l2_fast=2, l3_fast=30, l1_slow=200, l2_slow=2, l3_slow=30, kama_trend_period=5):
    """
    Compute a market regime indicator based on the difference between two KAMA (fast and slow).

    This function calculates two KAMA indicators using two different parameter sets (fast and slow).
    It then derives a market regime signal based on the difference between the fast KAMA and slow KAMA:

    - Returns 1 when the fast KAMA is above the slow KAMA (bullish regime).
    - Returns -1 when the fast KAMA is below the slow KAMA (bearish regime).

    Parameters
    ----------
    df : polars.DataFrame
        DataFrame containing the input price series.
    col : str
        Column name on which to apply the KAMA calculation (e.g., 'close').
    l1_fast : int, optional
        Efficiency ratio lookback window for the fast KAMA (default is 50).
    l2_fast : int, optional
        Fastest EMA constant for the fast KAMA (default is 2).
    l3_fast : int, optional
        Slowest EMA constant for the fast KAMA (default is 30).
    l1_slow : int, optional
        Efficiency ratio lookback window for the slow KAMA (default is 200).
    l2_slow : int, optional
        Fastest EMA constant for the slow KAMA (default is 2).
    l3_slow : int, optional
        Slowest EMA constant for the slow KAMA (default is 30).

    Returns
    -------
    polars.Series
        A Series containing the market regime indicator:
        - 1 for bullish regime
        - -1 for bearish regime
    """

    # Calculate both KAMA values
    kama_fast = trend.kama(df, col, l1=l1_fast, l2=l2_fast, l3=l3_fast)
    kama_slow = trend.kama(df, col, l1=l1_slow, l2=l2_slow, l3=l3_slow)

    # Difference & regime detection
    kama_diff_pct = (kama_fast - kama_slow)/kama_slow * 100

    #    kama_trend = np.sign(kama_diff)
    kama_trend_slow = talib.LINEARREG_ANGLE(kama_slow, kama_trend_period)
    kama_trend_fast = talib.LINEARREG_ANGLE(kama_fast, kama_trend_period)

    return kama_fast, kama_slow, kama_diff_pct, kama_trend_slow, kama_trend_fast


def displacement_detection(df, type_range="standard", strength=3, period=100):
    """
    This function calculates and adds a 'displacement' column to a provided DataFrame. Displacement is determined based on
    the 'candle_range' which is calculated differently according to the 'type_range' parameter. Then, it calculates the
    standard deviation of the 'candle_range' over a given period and sets a 'threshold'. If 'candle_range' exceeds this 'threshold',
    a displacement is detected and marked as 1 in the 'displacement' column.

    Parameters:
    df (pd.DataFrame): The DataFrame to add the columns to. This DataFrame should have 'open', 'close', 'high', and 'low' columns.
    type_range (str, optional): Defines how to calculate 'candle_range'. 'standard' calculates it as the absolute difference between
                                'close' and 'open', 'extremum' calculates it as the absolute difference between 'high' and 'low'.
                                Default is 'standard'.
    strengh (int, optional): The multiplier for the standard deviation to set the 'threshold'. Default is 3.
    period (int, optional): The period to use for calculating the standard deviation. Default is 100.

    Returns:
    pd.DataFrame: The original DataFrame, but with four new columns: 'candle_range', 'MSTD', 'threshold' and 'displacement'.

    Raises:
    ValueError: If an unsupported 'type_range' is provided.
    """
#    df = df.copy()

    # Choose your type_range
    if type_range == "standard":
        candle_range = np.abs(df['Close'] - df['Open'])
    elif type_range == "extremum":
        candle_range = df['High'] - df['Low']
    else:
        raise ValueError("Select correct range type")

    # Compute the STD of the candle range
    candle_range_std = np.maximum(candle_range.rolling(period).std(), df['Close']*0.0001)
    threshold = candle_range_std * strength

    # Displacement if the candle range is above the threshold
    displacement = np.where(candle_range>threshold, np.sign(df['Close'] - df['Open']) * np.log1p(candle_range / threshold), 0)      # np.sign(df['Close'] - df['Open'])

    # TODO: # Shift by one because we only know that we have a displacement at the end of the candle (BE CAREFUL)
    #    df["green_displacement"] = df["green_displacement"].shift(1)
    #    df["red_displacement"] = df["red_displacement"].shift(1)

    #    df["high_displacement"] = np.nan
    #     df["low_displacement"] = np.nan

    up_displacement_high = np.where(displacement == 1, df['High'], 0)
    up_displacement_low = np.where(displacement == 1, df['Low'], 0)
    
    down_displacement_low  = np.where(displacement == -1, df['Low'], 0)
    down_displacement_high  = np.where(displacement == -1, df['High'], 0)


    return candle_range, candle_range_std, displacement, up_displacement_high, up_displacement_low, down_displacement_high, down_displacement_low


def gap_detection(df, lookback=2):
    """
    Detects and calculates the bullish and bearish gaps in the given DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with columns 'high' and 'low' representing the high and low prices for each period.
    - lookback (int, optional): Number of periods to look back to detect gaps. Default is 2.

    Returns:
    - pd.DataFrame: DataFrame with additional columns:
        * 'bullish_gap_sup': Upper boundary of the bullish gap.
        * 'bullish_gap_inf': Lower boundary of the bullish gap.
        * 'bearish_gap_sup': Upper boundary of the bearish gap.
        * 'bearish_gap_inf': Lower boundary of the bearish gap.
        * 'bullish_gap_size': Size of the bullish gap.
        * 'bearish_gap_size': Size of the bearish gap.

    The function first identifies the bullish and bearish gaps by comparing the current period's high/low prices
    with the high/low prices of the lookback period. It then calculates the size of each gap and forward-fills any
    missing values in the gap boundaries.
    """

    bullish_gap = np.where(df['High'].shift(lookback) < df['Low'], 1, 0)
    bullish_gap_high = np.where(bullish_gap==1, df['Low'], 0)
    bullish_gap_low = np.where(bullish_gap==1, df['High'].shift(lookback), 0)

    bearish_gap = np.where(df['High'] < df['Low'].shift(lookback), -1, 0)
    bearish_gap_high = np.where(bearish_gap==-1, df['Low'].shift(lookback), 0)
    bearish_gap_low = np.where(bearish_gap==-1, df['High'], 0)

    bullish_gap_size = bullish_gap_high-bullish_gap_low
    bearish_gap_size = bearish_gap_high-bearish_gap_low

    return bullish_gap, bullish_gap_low, bullish_gap_high, bullish_gap_size, bearish_gap, bearish_gap_low, bearish_gap_high, bearish_gap_size


# def dc_event(Pt, Pext, threshold):
#     """
#     Compute if we have a POTENTIAL DC event
#     """
#     var = (Pt - Pext) / Pext
#
#     if var >= threshold:
#         return 1
#     if var <= -threshold:
#         return -1
#
#     return 0


def calculate_dc(df, col_close="Close", col_high="High", col_low="Low", threshold=0.02):
    """
    Compute the start and the end of a DC event
    """
    dfc = df.iloc[:, 0:4].reset_index(drop=True)

    # Initialize lists to store DC and OS events
    dc_events_up = []
    dc_events_down = []
    dc_events = []

    # Initialize the first DC event
    last_dc_price = dfc[col_close].iloc[0]
    last_dc_direction = 0  # +1 for up, -1 for down
    up_down_alternating = False

    # Initialize the current Min & Max for the OS events
    min_price = last_dc_price
    max_price = last_dc_price
    idx_min = 0
    idx_max = 0
    last_index = 0

    # Iterate over the price list
    for row in dfc.itertuples():

        # Uplate min & max prices
        h = getattr(row, col_high)
        l = getattr(row, col_low)
        c = getattr(row, col_close)
        if h > max_price:
            max_price = h
            idx_max = row.Index
        if l < min_price:
            min_price = l
            idx_min = row.Index
            
        #max_price = dfc[col_high].iloc[last_index:i].max()
        #min_price = dfc[col_low].iloc[last_index:i].min()
        #idx_min = dfc[col_high].iloc[last_index:i].idxmin()
        #idx_max = dfc[col_low].iloc[last_index:i].idxmax()

        
        # Add the DC event with the right index in the opposite direction
        if ((c-min_price)/min_price>=threshold) and (last_dc_direction != 1): #(dc_price_min == 1):
            dc_events_up.append([idx_min, row.Index, (c-min_price)/min_price])
            dc_events.append([idx_min, row.Index, (c-min_price)/min_price])
            if up_down_alternating:
                last_dc_direction = 1
            last_index=row.Index
            max_price = h
            min_price = l
            idx_min = last_index
            idx_max = last_index

        if ((max_price-c)/max_price>=threshold) and (last_dc_direction != -1): #(dc_price_max == -1):
            dc_events_down.append([idx_max, row.Index, (c-max_price)/max_price])
            dc_events.append([idx_max, row.Index, (c-max_price)/max_price])
            if up_down_alternating:
                last_dc_direction = -1
            last_index=row.Index
            max_price = h
            min_price = l
            idx_min = last_index
            idx_max = last_index

    return dc_events_up, dc_events_down, dc_events


# def calculate_trend(df, dc_events_down, dc_events_up):
#     """
#     Compute the DC + OS period (trend) using the DC event lists
#     """

#     # Initialize the variables
#     trend_events_up = []
#     trend_events_down = []
#     len_events_down = len(dc_events_down)
#     len_events_up = len(dc_events_up)

#     # Verify which event occured first (upward or downward movement)

#     # If the first event is a downward event
#     if dc_events_down[0][0] < dc_events_up[0][0]:

#         # Iterate on the index
#         for i in range(len_events_down):
#             # Calculate the start and end for each trend
#             if i == len_events_up:
#                 trend_events_down.append([dc_events_down[i][1], len(df) - 1])
#                 break
#             else:
#                 trend_events_down.append([dc_events_down[i][1], dc_events_up[i][0]])

#             if i == len_events_down - 1:
#                 trend_events_up.append([dc_events_up[i][1], len(df) - 1])
#             else:
#                 trend_events_up.append([dc_events_up[i][1], dc_events_down[i + 1][0]])

#     # If the first event is a upward event
#     else:

#         # Iterate on the index
#         for i in range(len_events_up):
#             # Calculate the start and end for each trend
#             if i == len_events_down:
#                 trend_events_up.append([dc_events_up[i][1], len(df) - 1])
#                 break
#             else:
#                 trend_events_up.append([dc_events_up[i][1], dc_events_down[i][0]])

#             if i == len_events_up - 1:
#                 trend_events_down.append([dc_events_down[i][1], len(df) - 1])
#             else:
#                 trend_events_down.append([dc_events_down[i][1], dc_events_up[i + 1][0]])

#     return trend_events_down, trend_events_up


def get_dc_price(df, dc_events, col_close="Close"):
    dc_events_prices = []
    for event in dc_events:
        prices = [df[col_close].iloc[event[0]], df[col_close].iloc[event[1]]]
        dc_events_prices.append(prices)
    return dc_events_prices


def DC_market_regime(df, col_close="Close", col_high="High", col_low="Low", threshold=0.01):
    """
    Determines the market regime based on Directional Change (DC) and trend events.

    Parameters:
    -----------
    df : polars.DataFrame
        A DataFrame containing financial data. The DataFrame should contain a 'close' column
        with the closing prices, and 'high' and 'low' columns for high and low prices.

    threshold : float
        The percentage threshold for DC events.

    Returns:
    --------
    df_copy : polars.DataFrame
        A new DataFrame containing the original data and a new column "market_regime",
        which indicates the market regime at each timestamp. A value of 1 indicates
        an upward trend, and a value of 0 indicates a downward trend.

    """
    #    df_copy = df.copy()

    # Extract DC and Trend events
    dc_events_up, dc_events_down, dc_events = calculate_dc(df, col_close=col_close, col_high=col_high, col_low=col_low, threshold=threshold)
    #print(len(dc_events_up), len(dc_events_down), len(dc_events))

#    trend_events_down, trend_events_up = calculate_trend(df, dc_events_down, dc_events_up)
    #    print(trend_events_up)
    #    print(trend_events_down)

    market_regime = np.zeros(len(df))

    for event in dc_events_up:
        # market_regime[event[0]]=1              # lookahead bias
        market_regime[event[1]]=event[2]*100
    for event in dc_events_down:
        # market_regime[event[0]]=-1              # lookahead bias
        market_regime[event[1]]=event[2]*100
    
#    for event in trend_events_up:
#        market_regime.iloc[event[0]]=0.75
#    for event in trend_events_down:
#        market_regime.iloc[event[0]]=0.25

    return market_regime

               

### Overlap

def overlap(df, col_close="Close", col_high="High", col_low="Low", bb_periods=12, bb_nbdev=2, ema1_period=20, ema2_period=50, sma1_period=14, sma2_period=30, sar_acc=0.02, sar_max=0.01, midprice_window=14):
    
    upperband, middleband, lowerband = talib.BBANDS(df[col_close], timeperiod=bb_periods, nbdevup=bb_nbdev, nbdevdn=bb_nbdev)

    ema1 = talib.EMA(df[col_close], timeperiod=ema1_period)

    ema2 = talib.EMA(df[col_close], timeperiod=ema2_period)

    sma1 = talib.SMA(df[col_close], timeperiod=sma1_period)

    sma2 = talib.SMA(df[col_close], timeperiod=sma2_period)

#    real = talib.MIDPOINT(df[col_close], timeperiod=14)
#    df['midpoint'] = real.astype('float16')

    midprice = talib.MIDPRICE(df[col_high], df[col_low], timeperiod=midprice_window)

    sar = talib.SAR(df[col_high], df[col_low], acceleration=sar_acc, maximum=sar_max)

    return upperband, middleband, lowerband, ema1, ema2, sma1, sma2, midprice, sar

def momentum(df, col_close="Close", col_high="High", col_low="Low", rsi_period=14, macd_fast=9, macd_slow=26, macd_signal=12, stoch_fastk=5, stoch_slowk=9, stoch_slowd=14):
#    real = talib.ROCP(df.Close, timeperiod=1)
#    df['rClose'] = real.astype('float16')

#    real = talib.ADX(df.High, df.Low, df.Close, timeperiod=14)
#    df['adx'] = real.astype('float16')

#    real = talib.ADXR(df.High, df.Low, df.Close, timeperiod=14)
#    df['adxr'] = real.astype('float16')

    cci = talib.CCI(df.High, df.Low, df.Close, timeperiod=14)

    macd, macdsignal, macdhist = talib.MACDEXT(df.Close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)

    ppo = talib.PPO(df.Close, fastperiod=12, slowperiod=26, matype=0)

    minus_di = talib.MINUS_DI(df.High, df.Low, df.Close, timeperiod=14)
    plus_di = talib.PLUS_DI(df.High, df.Low, df.Close, timeperiod=14)

    rsi = talib.RSI(df.Close, timeperiod=rsi_period)

    slowk, slowd = talib.STOCH(df.High, df.Low, df.Close, fastk_period=stoch_fastk, slowk_period=stoch_slowk,
                               slowk_matype=0, slowd_period=stoch_slowd, slowd_matype=0)

    fastk, fastd = talib.STOCHRSI(df.Close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)

    willr = talib.WILLR(df.High, df.Low, df.Close, timeperiod=14)/50
    df['willr'] = (willr + 1).astype('float16')

    copp = talib.WMA(talib.ROC(df.Close, timeperiod=14) + talib.ROC(df.Close, timeperiod=11), timeperiod=10)  # coppock
    df['copp'] = copp.astype('float16')


def ichimoku(df, col_high="High", col_low="Low", tenkan_window=9, kijun_window=26):
    tenkan_sen = (df[col_high].rolling(window=tenkan_window).max() + df[col_low].rolling(window=tenkan_window).min()) / 2
    kijun_sen = (df[col_high].rolling(window=kijun_window).max() + df[col_low].rolling(window=kijun_window).min()) / 2 
    return tenkan_sen, kijun_sen

## PCA

def calc_kernel_pca(df, day_range, window, col_features, col_pca_comp):
    scaler = StandardScaler()
    # Call the PCA method from scikit learn
    num_components = len(col_pca_comp)
    pca = KernelPCA(n_components=num_components, kernel='rbf')
    # Process each date to create current and historical slices
    current_slice_scaled_pca_scores_df_sum = pd.DataFrame()
    for current_date_ind in range(window, len(day_range)):
        
        current_date = day_range[current_date_ind]
        current_date_window_start = day_range[current_date_ind-window]
        # Get data for current date
        current_slice = df[df.index.date == current_date]
        if current_slice.shape[0]==0:
            print(f"Current slice is empty for {current_date}")
            continue
        # Get data for previous 'window' days
        mask = (df.index.date < current_date) & (df.index.date >= current_date_window_start)
        historical_slice = df[mask]
    #    print(current_slice.shape)
    #    print(historical_slice.shape)
    
        # Standardize the features using the training set
        historical_slice_scaled = scaler.fit_transform(historical_slice[col_features])  # Fit on training data
        current_slice_scaled = scaler.transform(current_slice[col_features])
    
        # Train the PCA on the train set
        pca.fit(historical_slice_scaled)
        
        # Apply the PCA on the test dataset
        current_slice_scaled_pca_scores = pca.transform(current_slice_scaled)
    
        current_slice_scaled_pca_scores_df = pd.DataFrame(current_slice_scaled_pca_scores, columns = col_pca_comp, index=current_slice.index)
        current_slice_scaled_pca_scores_df['date'] = current_slice.index
        current_slice_scaled_pca_scores_df_sum = pd.concat([current_slice_scaled_pca_scores_df_sum, current_slice_scaled_pca_scores_df])
        current_slice_scaled_pca_scores_df_sum.set_index(['date'], inplace=True)
    return current_slice_scaled_pca_scores_df_sum


def market_regime_features(df, col_close="Close", col_high="High", col_low="Low",l1_fast=40,l2_fast=2,l3_fast=30,l1_slow=100,l2_slow=2,l3_slow=30, displacement_strength=3, market_regime_threshold=0.0015,
                           price_distribution_window_size=40,price_distribution_percentile_threshold=0.25, kama_trend_period=10,
                           ha_candle_period=10, dc_market_regime_period=20, displacement_strength_period=40, displacement_hull_period=30, displacement_sma_period=20,
                           displacement_hull_slope_period=10, gap_lookback=2, gap_hull_period=30, gap_hull_slope_period=20, ha_sign_ma_period=10):
    #print(datetime.now().strftime('%H:%M:%S'))

    df["adf_stat"], df["adf_pvalue"] = math.adf_test(df, col=col_close, window_size=40, lags=10, regression="ct")        #10s

    df['dc_market_regime'] = DC_market_regime(df, col_close=col_close, col_high=col_high, col_low=col_low, threshold=market_regime_threshold)        #40s
    df['dc_market_regime_ema'] = talib.EMA(df.loc[:,"dc_market_regime"], dc_market_regime_period)
    df['dc_market_regime_wma'] = talib.WMA(
                2 * talib.WMA(df.loc[:,"dc_market_regime"], dc_market_regime_period // 2) - talib.WMA(df.loc[:,"dc_market_regime"], dc_market_regime_period),
                int(round(np.sqrt(dc_market_regime_period)))
    )

    #log_wma = np.log1p(np.abs(df["dc_market_regime_wma"]))*np.sign(df["dc_market_regime_wma"])
    df['dc_market_regime_ema_log'] = np.log1p(np.abs(df["dc_market_regime_ema"]))*np.sign(df["dc_market_regime_ema"])

    df['kama_regime_fast'], df['kama_regime_slow'], df['kama_diff'], df['kama_trend_slow'], df['kama_trend_fast'] = kama_market_regime(df, col_close,
                                                                                                           l1_fast=l1_fast,
                                                                                                           l2_fast=l2_fast,
                                                                                                           l3_fast=l3_fast,
                                                                                                           l1_slow=l1_slow,
                                                                                                           l2_slow=l2_slow,
                                                                                                           l3_slow=l3_slow,
                                                                                                           kama_trend_period=kama_trend_period)        #1s

    df['bullish_gap'], df['bullish_gap_low'], df['bullish_gap_high'], df['bullish_gap_size'], df['bearish_gap'], df['bearish_gap_low'], df['bearish_gap_high'], df['bearish_gap_size'] = gap_detection(df, lookback=gap_lookback)        #1s
# Replacement with hull
    df['gap_hull'] = talib.WMA(
                2 * talib.WMA(df["bullish_gap"]+df["bearish_gap"], gap_hull_period // 2) - talib.WMA(df["bullish_gap"]+df["bearish_gap"], gap_hull_period),
                int(round(np.sqrt(gap_hull_period))),
    )
    df['gap_hull_slope'] = talib.LINEARREG_ANGLE(df['gap_hull'], gap_hull_slope_period)
#    df['gap_ema'] = talib.EMA(df.loc[:,"bullish_gap"]+df.loc[:,"bearish_gap"], gap_ema_period)*2
#    df['gap_ema_slope'] = talib.LINEARREG_ANGLE(df['gap_hull'], gap_ema_slope_period)
#

    
    df['candle_range'], df['candle_range_std'], df['displacement'], df['up_displacement_high'], df['up_displacement_low'], df['down_displacement_high'], df['down_displacement_low'] \
        = displacement_detection(df, type_range="standard", strength=displacement_strength, period=displacement_strength_period)        #1s

# Replacement with hull
    df['displacement_hull'] = talib.WMA(
                2 * talib.WMA(df["displacement"], displacement_hull_period // 2) - talib.WMA(df["displacement"], displacement_hull_period),
                int(round(np.sqrt(displacement_hull_period)))
    )
    df['displacement_hull_slope'] = talib.LINEARREG_ANGLE(df['displacement_hull'], displacement_hull_slope_period)
#
#    df['displacement_ema'] = talib.EMA(df.loc[:,"displacement"], displacement_ema_period)*2
#    df['displacement_sma'] = talib.SMA(df.loc[:,"displacement"], displacement_sma_period)*4
#    df['displacement_hull_slope'] = talib.LINEARREG_ANGLE(df['displacement_ema']+df['displacement_sma'], displacement_hull_slope_period)

    
# Replacement with hull
    df['ema_ha_candle_fill'] = talib.WMA(
                2 * talib.WMA(df["ha_candle_fill"], ha_candle_period // 2) - talib.WMA(df["ha_candle_fill"], ha_candle_period),
                int(round(np.sqrt(ha_candle_period))),
    )
    df['ema_ha_wickstrength'] = talib.WMA(
                2 * talib.WMA(df["ha_wickstrength"], ha_candle_period // 2) - talib.WMA(df["ha_wickstrength"], ha_candle_period),
                int(round(np.sqrt(ha_candle_period))),
    )
    df['ema_ha_sign'] = talib.WMA(
                2 * talib.WMA(df["ha_sign"], ha_sign_ma_period // 2) - talib.WMA(df["ha_sign"], ha_sign_ma_period),
                int(round(np.sqrt(ha_sign_ma_period))),
    )

    # Compute the percentage of closing prices in different range zones
#    df["0_to_25"] = candle.price_distribution(df, col=col_close, window_size=price_distribution_window_size, start_percentage=0.0, end_percentage=price_distribution_percentile_threshold)        #1s
#    df["25_to_75"] = candle.price_distribution(df, col=col_close, window_size=price_distribution_window_size, start_percentage=price_distribution_percentile_threshold, end_percentage=1-price_distribution_percentile_threshold)
#    df["75_to_100"] = candle.price_distribution(df, col=col_close, window_size=price_distribution_window_size, start_percentage=1-price_distribution_percentile_threshold, end_percentage=1.0)

    #print(datetime.now().strftime('%H:%M:%S'))
    return df


def numpy_fill(arr):
    '''Solution provided by Divakar.'''
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:,None], idx]
    return out

def exponential_growth(c, k, t):
    """Calculate the amount after a certain time with exponential growth."""
    return c * exp(k * t)
def exponential_decay(c, k, t):
    """Calculate the amount after a certain time with exponential decay."""
    return c * exp(-k * t)

@numba.jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def heikenashi_open(ha_close, ha_open_0=0):
    n = len(ha_close)
    ha_open = np.full(n, ha_open_0, dtype=np.float32)
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2
    return ha_open


@numba.jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
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

# Elliot Wave Oscillator
def ewo(dataframe, sma1_length=5, sma2_length=35):
    sma1 = ta.EMA(dataframe, timeperiod=sma1_length)
    sma2 = ta.EMA(dataframe, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / dataframe["close"] * 100
    return smadif


# Chaikin Money Flow
# def chaikin_money_flow(dataframe, n=20, fillna=False) -> Series:
#     """Chaikin Money Flow (CMF)
#     It measures the amount of Money Flow Volume over a specific period.
#     http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf
#     Args:
#         dataframe(polars.Dataframe): dataframe containing ohlcv
#         n(int): n period.
#         fillna(bool): if True, fill nan values.
#     Returns:
#         polars.Series: New feature generated.
#     """
#     df = dataframe.copy()
#     mfv = (df["close"] - df["low"] - (df["high"] - df["close"])) / (df["high"] - df["low"])
#     mfv = mfv.fillna(0.0)  # float division by zero
#     mfv *= df["volume"]
#     cmf = mfv.rolling(n, min_periods=0).sum() / df["volume"].rolling(n, min_periods=0).sum()
#     if fillna:
#         cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
#     return Series(cmf, name="cmf")


def tsi(dataframe: DataFrame, window_slow: int, window_fast: int, fillna=False) -> Series:
    """
    Indicator: True Strength Index (TSI)
    :param dataframe: DataFrame The original OHLC dataframe
    :param window_slow: slow smoothing period
    :param window_fast: fast smoothing period
    :param fillna: If True fill NaN values
    """
    df = dataframe.copy()
    min_periods_slow = 0 if fillna else window_slow
    min_periods_fast = 0 if fillna else window_fast
    close_diff = df["close"].diff()
    close_diff_abs = close_diff.abs()
    smooth_close_diff = (
        close_diff.ewm(span=window_slow, min_periods=min_periods_slow, adjust=False)
        .mean()
        .ewm(span=window_fast, min_periods=min_periods_fast, adjust=False)
        .mean()
    )
    smooth_close_diff_abs = (
        close_diff_abs.ewm(span=window_slow, min_periods=min_periods_slow, adjust=False)
        .mean()
        .ewm(span=window_fast, min_periods=min_periods_fast, adjust=False)
        .mean()
    )
    tsi = smooth_close_diff / smooth_close_diff_abs * 100
    if fillna:
        tsi = tsi.replace([np.inf, -np.inf], np.nan).fillna(0)
    return tsi


# Williams %R

def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
    of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
    Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
    of its recent trading range.
    The oscillator is on a negative scale, from âˆ’100 (lowest) up to 0 (highest).
    """
    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()
    WR = Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name="{0} Williams %R".format(period),
    )
    return WR * -100


# Volume Weighted Moving Average

def vwma(dataframe: DataFrame, length: int = 10):
    """Indicator: Volume Weighted Moving Average (VWMA)"""
    # Calculate Result
    pv = dataframe["close"] * dataframe["volume"]
    vwma = Series(ta.SMA(pv, timeperiod=length) / ta.SMA(dataframe["volume"], timeperiod=length))
    return vwma


# Modified Elder Ray Index

def moderi(dataframe: DataFrame, len_slow_ma: int = 32) -> Series:
    slow_ma = Series(ta.EMA(vwma(dataframe, length=len_slow_ma), timeperiod=len_slow_ma))
    return slow_ma >= slow_ma.shift(1)  # we just need true & false for ERI trend


# zlema

def zlema(dataframe, timeperiod):
    lag = int(math.floor((timeperiod - 1) / 2))
    if isinstance(dataframe, Series):
        ema_data = dataframe + (dataframe - dataframe.shift(lag))
    else:
        ema_data = 2*dataframe["close"] - dataframe["close"].shift(lag)
    return ta.EMA(ema_data, timeperiod=timeperiod)


# zlhull

def zlhull(dataframe, timeperiod):
    lag = int(math.floor((timeperiod - 1) / 2))
    if isinstance(dataframe, Series):
        wma_data = dataframe + (dataframe - dataframe.shift(lag))
    else:
        wma_data = 2*dataframe["close"] - dataframe["close"].shift(lag)
    return ta.WMA(
        2 * ta.WMA(wma_data, int(math.floor(timeperiod / 2))) - ta.WMA(wma_data, timeperiod),
        int(round(np.sqrt(timeperiod))),
    )


# hull

def hull(dataframe, timeperiod):
    if isinstance(dataframe, Series):
        return ta.WMA(
            2 * ta.WMA(dataframe, int(math.floor(timeperiod / 2))) - ta.WMA(dataframe, timeperiod),
            int(round(np.sqrt(timeperiod))),
        )
    else:
        return ta.WMA(
            2 * ta.WMA(dataframe["close"], int(math.floor(timeperiod / 2)))
            - ta.WMA(dataframe["close"], timeperiod),
            int(round(np.sqrt(timeperiod))),
        )


def kalman_filter(data, initial_state):
    kf = KalmanFilter(transition_matrices = [1],
              observation_matrices = [1],
              initial_state_mean = initial_state,
              initial_state_covariance = 1,
              observation_covariance=1,
              transition_covariance=.01)
    filtered, _ = kf.filter(data)   #kf.filter(data.fillna(initial_state))
    return filtered
