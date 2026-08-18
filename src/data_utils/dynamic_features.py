import numpy as np
import pandas as pd
import talib

from src.data_utils.features import ichimoku, market_regime_features, overlap


def dynamic_features(df, parameters, scaler, col_close="close", col_high="high", col_low="low") -> pd.DataFrame:

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
    df['kama_diff'] = df['kama_diff']/scaler

    df['kama_trend_slow_diff'] = df['kama_trend_slow'].diff()
    df['kama_trend_fast_diff'] = df['kama_trend_fast'].diff()
    df['kama_trend_slow_diff2'] = df['kama_trend_slow_diff'].diff()
    df['kama_trend_fast_diff2'] = df['kama_trend_fast_diff'].diff()

    df['ema_ha_wickstrength'] = df['ema_ha_wickstrength']/scaler

    df.loc[:,'atr'] = talib.ATR(df[col_high], df[col_low], df[col_close], timeperiod=parameters['atr_period'])/scaler/10

    df.loc[:,'upperband'], df.loc[:,'middleband'], df.loc[:,'lowerband'], df.loc[:,'ema1'], df.loc[:,'ema2'], df.loc[:,'sma1'], df.loc[:,'sma2'], df.loc[:,'midprice'], df.loc[:,'sar'] \
        = overlap(df, col_close=col_close, col_high=col_high, col_low=col_low, bb_periods=parameters['bb_periods'], bb_nbdev=parameters['bb_nbdev'], ema1_period=parameters['ema1_period'], sma1_period=parameters['sma1_period'], sma2_period=parameters['sma2_period'], sar_acc=parameters['sar_acc'], sar_max=parameters['sar_max'], midprice_window=parameters['midprice_window'])

    df.loc[:,'sma_cross'] = np.log(df['sma1']/df['sma2'])*100/scaler
    df.loc[:,'ema_cross'] = np.log(df['ema1']/df['ema2'])*100/scaler
    df.loc[:,'upperband_r'] = np.log(df['upperband']/df['ha_close'])*100/scaler
    df.loc[:,'middleband_r'] = np.log(df['middleband']/df['ha_close'])*100/scaler
    df.loc[:,'lowerband_r'] = np.log(df['lowerband']/df['ha_close'])*100/scaler
    df.loc[:,'sar_r'] = np.log(df['sar']/df['ha_close'])*100/scaler

    df.loc[:,'tenkan_sen'], df.loc[:,'kijun_sen'] = ichimoku(df, col_high="ha_high", col_low="ha_low", tenkan_window=parameters['tenkan_window'], kijun_window=parameters['kijun_window'])

    df.loc[:,'r_tenkan_sen'] = np.log(df['tenkan_sen']/df['ha_close'])*200/scaler
    df.loc[:,'r_kijun_sen'] = np.log(df['kijun_sen']/df['ha_close'])*200/scaler

    # momentum

    df.loc[:,'cci'] = talib.CCI(df[col_high], df[col_low], df[col_close], timeperiod=parameters['cci_timeperiods'])/100
    df.loc[:,'cci_ha'] = talib.CCI(df['ha_high'], df['ha_low'], df['ha_close'], timeperiod=parameters['cci_timeperiods'])/100


    df.loc[:,'macd'], df.loc[:,'macdsignal'], df.loc[:,'macdhist'] = talib.MACDEXT(df[col_close],
                                                                 fastperiod=parameters['macd_fastperiod'], fastmatype=0,
                                                                 slowperiod=parameters['macd_slowperiod'], slowmatype=0,
                                                                 signalperiod=parameters['macd_signalperiod'], signalmatype=0)
    df.loc[:,'macd_slope'] = talib.LINEARREG_ANGLE(df['macd'], parameters['macd_slope_period'])
    df['macdhist'] = df['macdhist']/scaler/10

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

    df.loc[:,'adx'] = talib.ADX(df[col_high],df[col_low],df[col_close],timeperiod=parameters['adx_timeperiod'])/50-1
    df.loc[:,'di_plus'] = talib.PLUS_DI(df[col_high],df[col_low],df[col_close],timeperiod=parameters['di_timeperiod'])/20-1
    df.loc[:,'di_minus'] = talib.MINUS_DI(df[col_high],df[col_low],df[col_close],timeperiod=parameters['di_timeperiod'])/20-1
    df.loc[:,'di_diff'] = df['di_plus'] - df['di_minus']

    # # Prepare data
    # wavelet_coeff = wavelet_transform(df['ha_close'], 8)
    # df.loc[:,'wavelet_reconstr'] = inverse_wavelet_transform(wavelet_coeff, 8, 4)

    return df

