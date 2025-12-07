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

Author: Computational Finance Project
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import from utils modules
from utils.data_loader import load_historical_data, load_all_historical as load_all_symbols
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
from utils.metrics import (
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_win_rate,
)


# =============================================================================
# MOMENTUM STRATEGIES
# =============================================================================

class MomentumStrategy:
    """Base class for momentum strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.signals = None
        self.positions = None
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals. Override in subclass."""
        raise NotImplementedError
        
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate strategy returns based on signals."""
        if self.signals is None:
            self.generate_signals(df)
            
        df = df.copy()
        df['signal'] = self.signals
        df['position'] = df['signal'].shift(1)  # Enter position next day
        df['market_return'] = df['adj_close'].pct_change()
        df['strategy_return'] = df['position'] * df['market_return']
        df['cumulative_market'] = (1 + df['market_return']).cumprod()
        df['cumulative_strategy'] = (1 + df['strategy_return']).cumprod()
        
        return df


class PriceMomentum(MomentumStrategy):
    """
    Classic Price Momentum Strategy. (long only)
    Buy when price is above its N-period moving average.
    Stronger signal when short MA crosses above long MA.
    """
    
    def __init__(self, short_window: int = 20, long_window: int = 50):
        super().__init__("Price Momentum")
        self.short_window = short_window
        self.long_window = long_window
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['sma_short'] = calculate_sma(df['adj_close'], self.short_window)
        df['sma_long'] = calculate_sma(df['adj_close'], self.long_window)
        
        # Signal: 1 (long) when short MA > long MA, 0 otherwise
        df['signal'] = np.where(df['sma_short'] > df['sma_long'], 1, 0)
        
        # Crossover detection
        df['crossover'] = df['signal'].diff()
        
        self.signals = df['signal']
        return df


class ROCMomentum(MomentumStrategy):
    """
    Rate of Change (ROC) Momentum Strategy.
    Buy when ROC is positive and above threshold.
    """
    
    def __init__(self, period: int = 12, threshold: float = 0.0):
        super().__init__("ROC Momentum")
        self.period = period
        self.threshold = threshold
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # ROC = (Current Price - Price n periods ago) / Price n periods ago × 100
        df['roc'] = ((df['adj_close'] - df['adj_close'].shift(self.period)) / 
                     df['adj_close'].shift(self.period)) * 100
        
        # Signal: 1 when ROC > threshold
        df['signal'] = np.where(df['roc'] > self.threshold, 1, 0)
        
        self.signals = df['signal']
        return df


class RSIMomentum(MomentumStrategy):
    """
    RSI-based Momentum Strategy.
    Long when RSI breaks out above oversold (crosses up from below 30 to above 30).
    Exit/Short when RSI breaks down from overbought (crosses down from above 70 to below 70).
    """
    
    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        super().__init__("RSI Momentum")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['rsi'] = calculate_rsi(df['adj_close'], self.period)
        
        # Signal logic:
        # Long (1) when RSI crosses UP through oversold (breakout)
        # Exit (0) when RSI crosses DOWN through overbought (breakdown)
        # Hold previous position otherwise
        df['signal'] = 0
        position = 0
        
        for i in range(1, len(df)):
            prev_rsi = df['rsi'].iloc[i - 1]
            curr_rsi = df['rsi'].iloc[i]
            
            # Breakout: RSI crosses up from below oversold to above oversold
            if prev_rsi < self.oversold and curr_rsi >= self.oversold:
                position = 1  # Long signal
            # Breakdown: RSI crosses down from above overbought to below overbought
            elif prev_rsi > self.overbought and curr_rsi <= self.overbought:
                position = 0  # Exit/Short signal
                
            df.loc[df.index[i], 'signal'] = position
        
        self.signals = df['signal']
        return df


class MACDMomentum(MomentumStrategy):
    """
    MACD-based Momentum Strategy.
    Buy when MACD line crosses above signal line.
    """
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__("MACD Momentum")
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(
            df['adj_close'], self.fast, self.slow, self.signal_period
        )
        
        # Signal: 1 when MACD > Signal line
        df['signal'] = np.where(df['macd'] > df['macd_signal'], 1, 0)
        
        self.signals = df['signal']
        return df


class VolumeWeightedMomentum(MomentumStrategy):
    """
    Volume-Weighted Momentum Strategy.
    Combines price momentum with volume confirmation.
    Strong momentum = Price up + Volume above average.
    """
    
    def __init__(self, price_period: int = 20, volume_period: int = 20, volume_factor: float = 1.5):
        super().__init__("Volume-Weighted Momentum")
        self.price_period = price_period
        self.volume_period = volume_period
        self.volume_factor = volume_factor
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Price momentum: current price vs SMA
        df['price_sma'] = calculate_sma(df['adj_close'], self.price_period)
        df['price_momentum'] = df['adj_close'] > df['price_sma']
        
        # Volume confirmation: volume above average
        df['volume_sma'] = calculate_sma(df['volume'], self.volume_period)
        df['volume_confirm'] = df['volume'] > df['volume_sma'] * self.volume_factor
        
        # Signal: 1 when both conditions met
        df['signal'] = np.where(df['price_momentum'] & df['volume_confirm'], 1, 0)
        
        self.signals = df['signal']
        return df


class OBVMomentum(MomentumStrategy):
    """
    On-Balance Volume Momentum Strategy.
    Buy when OBV trend is up (OBV > OBV SMA).
    """
    
    def __init__(self, obv_period: int = 20):
        super().__init__("OBV Momentum")
        self.obv_period = obv_period
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['obv'] = calculate_obv(df['adj_close'], df['volume'])
        df['obv_sma'] = calculate_sma(df['obv'], self.obv_period)
        
        # Signal: 1 when OBV > OBV SMA (accumulation)
        df['signal'] = np.where(df['obv'] > df['obv_sma'], 1, 0)
        
        self.signals = df['signal']
        return df


class VPTMomentum(MomentumStrategy):
    """
    Volume Price Trend Momentum Strategy.
    Combines price change and volume into a cumulative indicator.
    """
    
    def __init__(self, vpt_period: int = 20):
        super().__init__("VPT Momentum")
        self.vpt_period = vpt_period
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['vpt'] = calculate_vpt(df['adj_close'], df['volume'])
        df['vpt_sma'] = calculate_sma(df['vpt'], self.vpt_period)
        
        # Signal: 1 when VPT > VPT SMA
        df['signal'] = np.where(df['vpt'] > df['vpt_sma'], 1, 0)
        
        self.signals = df['signal']
        return df


class MFIMomentum(MomentumStrategy):
    """
    Money Flow Index Momentum Strategy.
    Volume-weighted RSI - buy in oversold, sell in overbought.
    """
    
    def __init__(self, period: int = 14, oversold: int = 20, overbought: int = 80):
        super().__init__("MFI Momentum")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df['mfi'] = calculate_mfi(df['high'], df['low'], df['adj_close'], 
                                  df['volume'], self.period)
        
        # Signal logic similar to RSI
        df['signal'] = 0
        position = 0
        
        for i in range(1, len(df)):
            if df['mfi'].iloc[i] < self.oversold:
                position = 1
            elif df['mfi'].iloc[i] > self.overbought:
                position = 0
            df.loc[df.index[i], 'signal'] = position
        
        self.signals = df['signal']
        return df


class DualMomentum(MomentumStrategy):
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


class TripleMomentum(MomentumStrategy):
    """
    Triple Momentum Strategy.
    Combines short, medium, and long-term momentum signals.
    All three must agree for a position.
    """
    
    def __init__(self, short: int = 21, medium: int = 63, long: int = 252):
        super().__init__("Triple Momentum")
        self.short = short
        self.medium = medium
        self.long = long
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Calculate returns over different periods
        df['ret_short'] = df['adj_close'].pct_change(self.short)
        df['ret_medium'] = df['adj_close'].pct_change(self.medium)
        df['ret_long'] = df['adj_close'].pct_change(self.long)
        
        # All three must be positive
        df['signal'] = np.where(
            (df['ret_short'] > 0) & 
            (df['ret_medium'] > 0) & 
            (df['ret_long'] > 0), 
            1, 0
        )
        
        self.signals = df['signal']
        return df


class AcceleratingMomentum(MomentumStrategy):
    """
    Accelerating Momentum Strategy.
    Looks for stocks where momentum is increasing (momentum of momentum).
    """
    
    def __init__(self, momentum_period: int = 20, acceleration_period: int = 5):
        super().__init__("Accelerating Momentum")
        self.momentum_period = momentum_period
        self.acceleration_period = acceleration_period
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # First-order momentum
        df['momentum'] = df['adj_close'].pct_change(self.momentum_period)
        
        # Second-order momentum (acceleration)
        df['acceleration'] = df['momentum'].diff(self.acceleration_period)
        
        # Signal: positive momentum AND positive acceleration
        df['signal'] = np.where(
            (df['momentum'] > 0) & (df['acceleration'] > 0), 
            1, 0
        )
        
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


# =============================================================================
# BACKTESTING
# =============================================================================

def backtest_strategy(df: pd.DataFrame, strategy: MomentumStrategy, 
                      initial_capital: float = 100000,
                      transaction_cost: float = 0.001) -> Dict:
    """
    Backtest a momentum strategy.
    
    Args:
        df: DataFrame with OHLCV data
        strategy: MomentumStrategy instance
        initial_capital: Starting capital
        transaction_cost: Transaction cost as percentage
        
    Returns:
        Dictionary with backtest results
    """
    result_df = strategy.generate_signals(df)
    result_df = strategy.calculate_returns(result_df)
    
    # Account for transaction costs
    result_df['trade'] = result_df['signal'].diff().abs()
    result_df['tc'] = result_df['trade'] * transaction_cost
    result_df['strategy_return_net'] = result_df['strategy_return'] - result_df['tc']
    result_df['cumulative_strategy_net'] = (1 + result_df['strategy_return_net']).cumprod()
    
    # Calculate metrics
    returns = result_df['strategy_return_net'].dropna()
    
    total_return = result_df['cumulative_strategy_net'].iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    
    # Max drawdown
    cumulative = result_df['cumulative_strategy_net']
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Win rate
    winning_trades = (returns > 0).sum()
    total_trades = (returns != 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Number of trades
    num_trades = result_df['trade'].sum() / 2
    
    results = {
        'strategy': strategy.name,
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'num_trades': num_trades,
        'final_value': initial_capital * (1 + total_return),
        'data': result_df
    }
    
    return results


def compare_strategies(df: pd.DataFrame, strategies: List[MomentumStrategy]) -> pd.DataFrame:
    """Compare multiple strategies on the same data."""
    results = []
    
    for strategy in strategies:
        result = backtest_strategy(df, strategy)
        results.append({
            'Strategy': result['strategy'],
            'Total Return': f"{result['total_return']:.2%}",
            'Annual Return': f"{result['annual_return']:.2%}",
            'Volatility': f"{result['volatility']:.2%}",
            'Sharpe Ratio': f"{result['sharpe_ratio']:.2f}",
            'Max Drawdown': f"{result['max_drawdown']:.2%}",
            'Win Rate': f"{result['win_rate']:.2%}",
            'Num Trades': int(result['num_trades'])
        })
    
    return pd.DataFrame(results)


__all__ = [
    'MomentumStrategy',
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
    'MomentumPortfolio',
    'backtest_strategy',
    'compare_strategies',
]
