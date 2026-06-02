import quantreo.target_engineering as te
#import quantreo.features_engineering as fe

import talib

import numpy as np
import pandas as pd
from functools import lru_cache


@lru_cache(maxsize=32)
def target(df, candles_lookahead=15, avg_period_candles=8):
    targetcalc = pd.DataFrame()
    
    targetcalc['Close'] = df['Close']
    #targetcalc['Close_target_ewm'] = targetcalc['Close'].ewm(span=avg_period_candles).mean().shift(-candles_lookahead+avg_period_candles//2)
    targetcalc['Close_target_ma'] = targetcalc['Close'].rolling(avg_period_candles).mean().shift(-candles_lookahead+avg_period_candles//2)
    targetcalc.loc[:,'Close_target_ma'].ffill(inplace=True)
    
    #data[t]['Target']=targetcalc['Close'].combine(ewm_target, lambda x1, x2: 2 if x1-x2>intraday_target_threshold else 1 if x2-x1>intraday_target_threshold else 0).astype(np.int16)    
    #Binary classification
    targetcalc['Close_target_ma_bin'] = targetcalc['Close'].combine(targetcalc['Close_target_ma'], lambda x1, x2: 1 if x2/x1>=1 else (0 if x2/x1<1 else 0.5)).astype(np.int16)
    intraday_target_threshold=0.005
    #targetcalc['Close_target_ma_threshold'] = targetcalc['Close'].combine(targetcalc['Close_target_ma'], lambda x1, x2: x2/x1 if x2/x1>=(1+intraday_target_threshold) or x2/x1<=(1-intraday_target_threshold) else 0.5))
    targetcalc['Close_target_ma_cont'] = targetcalc['Close'].combine(targetcalc['Close_target_ma'], lambda x1, x2: (np.exp(abs((x2/x1-1)*100))-1)*np.sign((x2/x1-1)))

    return targetcalc

#[(np.exp(abs(x/10))-1)*np.sign(x) for x in range(-10,11,1)]


@lru_cache(maxsize=4)
def target_returns(df):
    # Compute the future log return over 3 periods
    fut_ret = te.magnitude.future_returns(df, window_size=5, log_return=True, close_col="Close").rename('fut_ret')

    # Generate a directional label: 1 if return > 0, else 0
    direction = te.directional.future_returns_sign(df, window_size=5, close_col="Close").rename('direction')

    return fut_ret, direction

@lru_cache(maxsize=4)
def target_2barrier(df, tp=0.015, sl=-0.015):
    return te.directional.double_barrier_labeling(df, high_col="High", low_col="Low", open_col="Open", close_col="Close",
                                                             high_time_col="High_time", low_time_col="Low_time",
                                                             tp=tp, sl=sl, buy=True).rename('target_2barrier')

@lru_cache(maxsize=4)
def target_3barrier(dftp=0.015, sl=-0.015):
    return te.directional.triple_barrier_labeling(df, 30, high_col="High", low_col="Low", open_col="Open", close_col="Close",
                                                             high_time_col="High_time", low_time_col="How_time",
                                                             tp=tp, sl=sl, buy=True).rename('target_3barrier')

@lru_cache(maxsize=4)
def target_quantile(df, col='pct_change', upper_quantile_level=0.67, lower_quantile_level=0.33):
    return te.directional.quantile_label(df, col=col, upper_quantile_level=upper_quantile_level, lower_quantile_level=lower_quantile_level).rename('target_quantile')


