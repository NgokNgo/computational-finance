"""
Momentum Strategies using Price-Volume Data
============================================
This module implements various momentum-based trading strategies:

1. Price Momentum (Classic)
2. Volume-Weighted Momentum
3. Rate of Change (ROC)
4. Relative Strength Index (RSI)
5. Moving Average Convergence Divergence (MACD)
6. On-Balance Volume (OBV) Momentum
7. Volume Price Trend (VPT)
8. Money Flow Index (MFI)
9. Dual Momentum (Absolute + Relative)
10. Momentum with Volume Confirmation

"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from strategies.base import BaseStrategy
from utils.indicators import (
    calculate_returns,
    calculate_log_returns,
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_obv,
    calculate_vpt,
    calculate_mfi,
    calculate_atr,
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _apply_mode_filter(signals: pd.Series, mode: str) -> pd.Series:
    """
    Apply mode filtering to base signals (-1/0/1).
    
    Args:
        signals: Base signal series (-1/0/1)
        mode: 'long_only', 'short_only', or 'long_short'
    
    Returns:
        Filtered signal series based on mode
    """
    if mode == 'long_only':
        return signals.apply(lambda x: max(x, 0))
    elif mode == 'short_only':
        return signals.apply(lambda x: min(x, 0))
    return signals  # long_short: no filtering


# =============================================================================
# MOMENTUM STRATEGIES
# =============================================================================


class PriceMomentum(BaseStrategy):
    """
    Classic Price Momentum Strategy.
    Buy when short MA > long MA, Sell when short MA < long MA.
    """
    
    def __init__(self, short_window: int = 20, long_window: int = 50, mode: str = 'long_only'):
        super().__init__(f"Price Momentum ({mode})")
        self.short_window = short_window
        self.long_window = long_window
        self.mode = mode
        
        if mode not in ['long_only', 'short_only', 'long_short']:
            raise ValueError(f"Invalid mode: {mode}. Choose 'long_only', 'short_only', or 'long_short'")
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['sma_short'] = calculate_sma(df['adj_close'], self.short_window)
        df['sma_long'] = calculate_sma(df['adj_close'], self.long_window)
        
        df['signal'] = np.where(df['sma_short'] > df['sma_long'], 1, 
                               np.where(df['sma_short'] < df['sma_long'], -1, 0))
        df['signal'] = _apply_mode_filter(df['signal'], self.mode)
        df['crossover'] = df['signal'].diff()
        
        self.signals = df['signal']
        return df


class ROCMomentum(BaseStrategy):
    """
    Rate of Change (ROC) Momentum Strategy.
    Buy when ROC > threshold, Sell when ROC < -threshold.
    """
    
    def __init__(self, period: int = 12, threshold: float = 0.0, mode: str = 'long_only'):
        super().__init__(f"ROC Momentum ({mode})")
        self.period = period
        self.threshold = threshold
        self.mode = mode
        
        if mode not in ['long_only', 'short_only', 'long_short']:
            raise ValueError(f"Invalid mode: {mode}. Choose 'long_only', 'short_only', or 'long_short'")
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['roc'] = ((df['adj_close'] - df['adj_close'].shift(self.period)) / 
                     df['adj_close'].shift(self.period)) * 100
        df['signal'] = np.where(df['roc'] > self.threshold, 1,
                               np.where(df['roc'] < -self.threshold, -1, 0))
        df['signal'] = _apply_mode_filter(df['signal'], self.mode)
        
        self.signals = df['signal']
        return df


class RSIMomentum(BaseStrategy):
    """
    RSI-based Momentum Strategy with flexible positioning modes.
    Long when RSI breaks out above oversold (crosses up from below 30 to above 30).
    Short when RSI breaks down from overbought (crosses down from above 70 to below 70).
    """
    
    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70, mode: str = 'long_only'):
        super().__init__(f"RSI Momentum ({mode})")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.mode = mode
        
        if mode not in ['long_only', 'short_only', 'long_short']:
            raise ValueError(f"Invalid mode: {mode}. Choose 'long_only', 'short_only', or 'long_short'")
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['rsi'] = calculate_rsi(df['adj_close'], self.period)
        
        signals = [0]
        for i in range(1, len(df)):
            pos = 0
            if df['rsi'].iloc[i-1] < self.oversold and df['rsi'].iloc[i] >= self.oversold:
                pos = 1
            elif df['rsi'].iloc[i-1] > self.overbought and df['rsi'].iloc[i] <= self.overbought:
                pos = -1 if self.mode in ['short_only', 'long_short'] else 0
            signals.append(pos if self.mode == 'long_short' else max(pos, 0) if self.mode == 'long_only' else min(pos, 0))
        
        df['signal'] = signals
        self.signals = df['signal']
        return df


class MACDMomentum(BaseStrategy):
    """
    MACD-based Momentum Strategy.
    Buy when MACD > signal line, Sell when MACD < signal line.
    """
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, mode: str = 'long_only'):
        super().__init__(f"MACD Momentum ({mode})")
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
        self.mode = mode
        
        if mode not in ['long_only', 'short_only', 'long_short']:
            raise ValueError(f"Invalid mode: {mode}. Choose 'long_only', 'short_only', or 'long_short'")
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(
            df['adj_close'], self.fast, self.slow, self.signal_period
        )
        
        df['signal'] = np.where(df['macd'] > df['macd_signal'], 1,
                               np.where(df['macd'] < df['macd_signal'], -1, 0))
        df['signal'] = _apply_mode_filter(df['signal'], self.mode)
        
        self.signals = df['signal']
        return df


class VolumeWeightedMomentum(BaseStrategy):
    """
    Volume-Weighted Momentum Strategy.
    Combines price momentum with volume confirmation.
    Strong momentum = Price momentum & Volume above average.
    """
    
    def __init__(self, price_period: int = 20, volume_period: int = 20, volume_factor: float = 1.5, mode: str = 'long_only'):
        super().__init__(f"Volume-Weighted Momentum ({mode})")
        self.price_period = price_period
        self.volume_period = volume_period
        self.volume_factor = volume_factor
        self.mode = mode
        
        if mode not in ['long_only', 'short_only', 'long_short']:
            raise ValueError(f"Invalid mode: {mode}. Choose 'long_only', 'short_only', or 'long_short'")
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['price_sma'] = calculate_sma(df['adj_close'], self.price_period)
        df['volume_sma'] = calculate_sma(df['volume'], self.volume_period)
        vol_confirm = df['volume'] > df['volume_sma'] * self.volume_factor
        
        df['signal'] = np.where((df['adj_close'] > df['price_sma']) & vol_confirm, 1,
                               np.where((df['adj_close'] < df['price_sma']) & vol_confirm, -1, 0))
        df['signal'] = _apply_mode_filter(df['signal'], self.mode)
        
        self.signals = df['signal']
        return df


class OBVMomentum(BaseStrategy):
    """
    On-Balance Volume Momentum Strategy.
    Buy when OBV > OBV SMA (accumulation), Sell when OBV < OBV SMA (distribution).
    """
    
    def __init__(self, obv_period: int = 20, mode: str = 'long_only'):
        super().__init__(f"OBV Momentum ({mode})")
        self.obv_period = obv_period
        self.mode = mode
        
        if mode not in ['long_only', 'short_only', 'long_short']:
            raise ValueError(f"Invalid mode: {mode}. Choose 'long_only', 'short_only', or 'long_short'")
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['obv'] = calculate_obv(df['adj_close'], df['volume'])
        df['obv_sma'] = calculate_sma(df['obv'], self.obv_period)
        
        df['signal'] = np.where(df['obv'] > df['obv_sma'], 1,
                               np.where(df['obv'] < df['obv_sma'], -1, 0))
        df['signal'] = _apply_mode_filter(df['signal'], self.mode)
        
        self.signals = df['signal']
        return df


class VPTMomentum(BaseStrategy):
    """
    Volume Price Trend Momentum Strategy.
    Buy when VPT > VPT SMA, Sell when VPT < VPT SMA.
    """
    
    def __init__(self, vpt_period: int = 20, mode: str = 'long_only'):
        super().__init__(f"VPT Momentum ({mode})")
        self.vpt_period = vpt_period
        self.mode = mode
        
        if mode not in ['long_only', 'short_only', 'long_short']:
            raise ValueError(f"Invalid mode: {mode}. Choose 'long_only', 'short_only', or 'long_short'")
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['vpt'] = calculate_vpt(df['adj_close'], df['volume'])
        df['vpt_sma'] = calculate_sma(df['vpt'], self.vpt_period)
        
        df['signal'] = np.where(df['vpt'] > df['vpt_sma'], 1,
                               np.where(df['vpt'] < df['vpt_sma'], -1, 0))
        df['signal'] = _apply_mode_filter(df['signal'], self.mode)
        
        self.signals = df['signal']
        return df


class MFIMomentum(BaseStrategy):
    """
    Money Flow Index Momentum Strategy with flexible positioning modes.
    Volume-weighted RSI - buy in oversold, sell in overbought.
    """
    
    def __init__(self, period: int = 14, oversold: int = 20, overbought: int = 80, mode: str = 'long_only'):
        super().__init__(f"MFI Momentum ({mode})")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.mode = mode
        
        if mode not in ['long_only', 'short_only', 'long_short']:
            raise ValueError(f"Invalid mode: {mode}. Choose 'long_only', 'short_only', or 'long_short'")
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['mfi'] = calculate_mfi(df['high'], df['low'], df['adj_close'], df['volume'], self.period)
        
        signals = [0]
        for i in range(1, len(df)):
            mfi = df['mfi'].iloc[i]
            pos = 1 if mfi < self.oversold else (-1 if mfi > self.overbought and self.mode in ['short_only', 'long_short'] else 0)
            signals.append(pos if self.mode == 'long_short' else max(pos, 0) if self.mode == 'long_only' else min(pos, 0))
        
        df['signal'] = signals
        self.signals = df['signal']
        return df


class DualMomentum(BaseStrategy):
    """
    Dual Momentum Strategy (Gary Antonacci).
    Combines Absolute Momentum (time-series) and Relative Momentum (cross-sectional).
    
    Absolute: Asset must have positive momentum (return > risk-free rate proxy)
    Relative: Asset must outperform benchmark or other assets
    """
    
    def __init__(self, lookback: int = 252, rf_proxy: float = 0.02):
        super().__init__("Dual Momentum")
        self.lookback = lookback
        self.rf_proxy = rf_proxy  # Annual risk-free rate proxy
        
    def generate_signals(self, df: pd.DataFrame, benchmark: pd.DataFrame = None) -> pd.DataFrame:
        df = df.copy()
        
        # Absolute Momentum: Is return > risk-free?
        df['return_lookback'] = df['adj_close'].pct_change(self.lookback)
        rf_period = self.rf_proxy * (self.lookback / 252)  # Adjusted for period
        df['abs_momentum'] = df['return_lookback'] > rf_period
        
        # Relative Momentum (if benchmark provided)
        if benchmark is not None:
            benchmark = benchmark.copy()
            benchmark['bench_return'] = benchmark['adj_close'].pct_change(self.lookback)
            # Merge on date
            df = df.merge(benchmark[['date', 'bench_return']], on='date', how='left')
            df['rel_momentum'] = df['return_lookback'] > df['bench_return']
            
            # Signal: both conditions must be true
            df['signal'] = np.where(df['abs_momentum'] & df['rel_momentum'], 1, 0)
        else:
            # Without benchmark, use only absolute momentum
            df['signal'] = np.where(df['abs_momentum'], 1, 0)
        
        self.signals = df['signal']
        return df


class TripleMomentum(BaseStrategy):
    """
    Triple Momentum Strategy.
    Combines short, medium, and long-term momentum signals.
    All three must agree for a position (all up or all down).
    """
    
    def __init__(self, short: int = 21, medium: int = 63, long: int = 252, mode: str = 'long_only'):
        super().__init__(f"Triple Momentum ({mode})")
        self.short = short
        self.medium = medium
        self.long = long
        self.mode = mode
        
        if mode not in ['long_only', 'short_only', 'long_short']:
            raise ValueError(f"Invalid mode: {mode}. Choose 'long_only', 'short_only', or 'long_short'")
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        r_s, r_m, r_l = df['adj_close'].pct_change(self.short), df['adj_close'].pct_change(self.medium), df['adj_close'].pct_change(self.long)
        df['signal'] = np.where((r_s > 0) & (r_m > 0) & (r_l > 0), 1,
                               np.where((r_s < 0) & (r_m < 0) & (r_l < 0), -1, 0))
        df['signal'] = _apply_mode_filter(df['signal'], self.mode)
        
        self.signals = df['signal']
        return df


class AcceleratingMomentum(BaseStrategy):
    """
    Accelerating Momentum Strategy.
    Looks for stocks where momentum is increasing/decreasing (momentum of momentum).
    """
    
    def __init__(self, momentum_period: int = 20, acceleration_period: int = 5, mode: str = 'long_only'):
        super().__init__(f"Accelerating Momentum ({mode})")
        self.momentum_period = momentum_period
        self.acceleration_period = acceleration_period
        self.mode = mode
        
        if mode not in ['long_only', 'short_only', 'long_short']:
            raise ValueError(f"Invalid mode: {mode}. Choose 'long_only', 'short_only', or 'long_short'")
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['momentum'] = df['adj_close'].pct_change(self.momentum_period)
        df['accel'] = df['momentum'].diff(self.acceleration_period)
        
        df['signal'] = np.where((df['momentum'] > 0) & (df['accel'] > 0), 1,
                               np.where((df['momentum'] < 0) & (df['accel'] < 0), -1, 0))
        df['signal'] = _apply_mode_filter(df['signal'], self.mode)
        
        self.signals = df['signal']
        return df


# =============================================================================
# PORTFOLIO CONSTRUCTION
# =============================================================================

class MomentumPortfolio:
    """
    Cross-sectional Momentum Portfolio.
    Ranks stocks by momentum and holds top performers.
    """
    
    def __init__(self, lookback: int = 252, holding_period: int = 21, 
                 top_n: int = 3, bottom_n: int = 0):
        self.lookback = lookback
        self.holding_period = holding_period
        self.top_n = top_n
        self.bottom_n = bottom_n  # For long-short
        
    def calculate_momentum_scores(self, symbols_data: Dict[str, pd.DataFrame], 
                                   date: pd.Timestamp) -> Dict[str, float]:
        """Calculate momentum score for each symbol at a given date."""
        scores = {}
        
        for symbol, df in symbols_data.items():
            df_filtered = df[df['date'] <= date].copy()
            if len(df_filtered) < self.lookback:
                continue
                
            # Momentum = return over lookback period
            current_price = df_filtered['adj_close'].iloc[-1]
            past_price = df_filtered['adj_close'].iloc[-self.lookback]
            
            if past_price > 0:
                scores[symbol] = (current_price - past_price) / past_price
                
        return scores
    
    def get_portfolio_weights(self, symbols_data: Dict[str, pd.DataFrame],
                               date: pd.Timestamp) -> Dict[str, float]:
        """Get portfolio weights based on momentum ranking."""
        scores = self.calculate_momentum_scores(symbols_data, date)
        
        if not scores:
            return {}
            
        # Sort by momentum
        sorted_symbols = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        weights = {}
        
        # Long top N
        top_symbols = sorted_symbols[:self.top_n]
        for symbol in top_symbols:
            weights[symbol] = 1.0 / self.top_n
            
        # Short bottom N (if specified)
        if self.bottom_n > 0:
            bottom_symbols = sorted_symbols[-self.bottom_n:]
            for symbol in bottom_symbols:
                weights[symbol] = -1.0 / self.bottom_n
                
        return weights


__all__ = [
    'PriceMomentum',
    'ROCMomentum',
    'RSIMomentum',
    'MACDMomentum',
    'VolumeWeightedMomentum',
    'OBVMomentum',
    'VPTMomentum',
    'MFIMomentum',
    'DualMomentum',
    'TripleMomentum',
    'AcceleratingMomentum',
    'MomentumPortfolio'
]
