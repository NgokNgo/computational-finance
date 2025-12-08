"""Technical Indicators Module.

Common technical indicators used in trading strategies.

Categories:
    - Basic: Returns, Moving Averages, Standard Deviation
    - Momentum: RSI, MACD, Stochastic
    - Volume: OBV, VPT, MFI
    - Volatility: ATR, Bollinger Bands
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple


# =============================================================================
# BASIC INDICATORS
# =============================================================================

def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate simple returns over specified periods."""
    return prices.pct_change(periods)


def calculate_log_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate log returns over specified periods."""
    return np.log(prices / prices.shift(periods))


def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return series.rolling(window=window).mean()


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def calculate_std(series: pd.Series, window: int) -> pd.Series:
    """Calculate Rolling Standard Deviation."""
    return series.rolling(window=window).std()


# =============================================================================
# MOMENTUM INDICATORS
# =============================================================================

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI).

    RSI = 100 - (100 / (1 + RS)) where RS = Average Gain / Average Loss.

    Args:
        prices: Price series.
        period: Lookback period (default 14).

    Returns:
        RSI values (0-100).
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, 
                   signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: Price series
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)
        
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                         k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator (%K and %D).
    
    Args:
        high, low, close: Price series
        k_period: %K lookback period
        d_period: %D smoothing period
        
    Returns:
        Tuple of (%K, %D)
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    
    return k, d


def calculate_roc(prices: pd.Series, period: int = 10) -> pd.Series:
    """
    Calculate Rate of Change (ROC).
    
    Args:
        prices: Price series
        period: Lookback period
        
    Returns:
        ROC values (percentage)
    """
    return (prices - prices.shift(period)) / prices.shift(period) * 100


def calculate_momentum(prices: pd.Series, period: int = 10) -> pd.Series:
    """
    Calculate Momentum indicator.
    
    Args:
        prices: Price series
        period: Lookback period
        
    Returns:
        Momentum values
    """
    return prices - prices.shift(period)


# =============================================================================
# VOLUME INDICATORS
# =============================================================================

def calculate_obv(prices: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).
    OBV accumulates volume based on price direction.
    
    Args:
        prices: Price series
        volume: Volume series
        
    Returns:
        OBV series
    """
    price_change = prices.diff()
    obv = pd.Series(index=prices.index, dtype=float)
    obv.iloc[0] = 0
    
    for i in range(1, len(prices)):
        if price_change.iloc[i] > 0:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif price_change.iloc[i] < 0:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv


def calculate_vpt(prices: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate Volume Price Trend (VPT).
    VPT = Previous VPT + Volume × (Today's Close − Previous Close) / Previous Close
    
    Args:
        prices: Price series
        volume: Volume series
        
    Returns:
        VPT series
    """
    price_change_pct = prices.pct_change()
    vpt = (volume * price_change_pct).cumsum()
    return vpt


def calculate_mfi(high: pd.Series, low: pd.Series, close: pd.Series, 
                  volume: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Money Flow Index (MFI) - Volume-weighted RSI.
    
    Args:
        high, low, close: Price series
        volume: Volume series
        period: Lookback period
        
    Returns:
        MFI values (0-100)
    """
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    
    tp_diff = typical_price.diff()
    
    positive_flow = pd.Series(0.0, index=close.index)
    negative_flow = pd.Series(0.0, index=close.index)
    
    positive_flow[tp_diff > 0] = raw_money_flow[tp_diff > 0]
    negative_flow[tp_diff < 0] = raw_money_flow[tp_diff < 0]
    
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()
    
    money_ratio = positive_mf / negative_mf
    mfi = 100 - (100 / (1 + money_ratio))
    
    return mfi


def calculate_ad_line(high: pd.Series, low: pd.Series, close: pd.Series, 
                       volume: pd.Series) -> pd.Series:
    """
    Calculate Accumulation/Distribution Line.
    
    Args:
        high, low, close: Price series
        volume: Volume series
        
    Returns:
        A/D Line series
    """
    clv = ((close - low) - (high - close)) / (high - low)
    clv = clv.fillna(0)
    ad = (clv * volume).cumsum()
    return ad


# =============================================================================
# VOLATILITY INDICATORS
# =============================================================================

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, 
                  period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR) for volatility.
    
    Args:
        high, low, close: Price series
        period: Lookback period
        
    Returns:
        ATR series
    """
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def calculate_bollinger_bands(prices: pd.Series, window: int = 20, 
                               num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        prices: Price series
        window: Moving average window
        num_std: Number of standard deviations
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle = calculate_sma(prices, window)
    std = calculate_std(prices, window)
    
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    
    return upper, middle, lower


def calculate_keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series,
                                ema_period: int = 20, atr_period: int = 10,
                                multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Keltner Channels.
    
    Args:
        high, low, close: Price series
        ema_period: EMA period for middle band
        atr_period: ATR period
        multiplier: ATR multiplier
        
    Returns:
        Tuple of (upper_channel, middle_channel, lower_channel)
    """
    middle = calculate_ema(close, ema_period)
    atr = calculate_atr(high, low, close, atr_period)
    
    upper = middle + (multiplier * atr)
    lower = middle - (multiplier * atr)
    
    return upper, middle, lower


def calculate_donchian_channels(high: pd.Series, low: pd.Series, 
                                 period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Donchian Channels.
    
    Args:
        high, low: Price series
        period: Lookback period
        
    Returns:
        Tuple of (upper_channel, middle_channel, lower_channel)
    """
    upper = high.rolling(window=period).max()
    lower = low.rolling(window=period).min()
    middle = (upper + lower) / 2
    
    return upper, middle, lower


__all__ = [
    # Basic
    'calculate_returns',
    'calculate_log_returns',
    'calculate_sma',
    'calculate_ema',
    'calculate_std',
    # Momentum
    'calculate_rsi',
    'calculate_macd',
    'calculate_stochastic',
    'calculate_roc',
    'calculate_momentum',
    # Volume
    'calculate_obv',
    'calculate_vpt',
    'calculate_mfi',
    'calculate_ad_line',
    # Volatility
    'calculate_atr',
    'calculate_bollinger_bands',
    'calculate_keltner_channels',
    'calculate_donchian_channels',
]
