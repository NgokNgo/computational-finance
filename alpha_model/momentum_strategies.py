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
import sys
import os
warnings.filterwarnings('ignore')

try:
    from alpha_model.utils import (
        # Data Loading
        load_historical_data,
        load_all_historical as load_all_symbols,
        # Technical Indicators
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
        # Performance Metrics
        calculate_sharpe_ratio,
        calculate_max_drawdown,
        calculate_win_rate,
    )
except (ImportError, ModuleNotFoundError):
    from .utils import (
        # Data Loading
        load_historical_data,
        load_all_historical as load_all_symbols,
        # Technical Indicators
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
        # Performance Metrics
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
    Classic Price Momentum Strategy.
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
    Buy when RSI crosses above oversold level, sell when crosses above overbought.
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
        # Buy (1) when RSI crosses above oversold
        # Sell (0) when RSI crosses above overbought
        # Hold previous position otherwise
        df['signal'] = 0
        position = 0
        
        for i in range(1, len(df)):
            if df['rsi'].iloc[i] < self.oversold:
                position = 1  # Buy signal
            elif df['rsi'].iloc[i] > self.overbought:
                position = 0  # Sell signal
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


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_strategy_performance(result: Dict, figsize: Tuple[int, int] = (14, 10)):
    """Plot strategy performance charts."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return
        
    df = result['data']
    
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    # Plot 1: Cumulative Returns
    axes[0].plot(df['date'], df['cumulative_market'], label='Buy & Hold', alpha=0.7)
    axes[0].plot(df['date'], df['cumulative_strategy_net'], label=result['strategy'], alpha=0.9)
    axes[0].set_title(f"{result['strategy']} - Cumulative Returns")
    axes[0].set_ylabel('Cumulative Return')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Price and Signals
    axes[1].plot(df['date'], df['adj_close'], label='Price', alpha=0.7)
    buy_signals = df[df['signal'].diff() == 1]
    sell_signals = df[df['signal'].diff() == -1]
    axes[1].scatter(buy_signals['date'], buy_signals['adj_close'], 
                   marker='^', color='green', label='Buy', s=50)
    axes[1].scatter(sell_signals['date'], sell_signals['adj_close'], 
                   marker='v', color='red', label='Sell', s=50)
    axes[1].set_title('Price with Buy/Sell Signals')
    axes[1].set_ylabel('Price')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Drawdown
    cumulative = df['cumulative_strategy_net']
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    axes[2].fill_between(df['date'], drawdown, 0, alpha=0.5, color='red')
    axes[2].set_title('Strategy Drawdown')
    axes[2].set_ylabel('Drawdown')
    axes[2].set_xlabel('Date')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_momentum_analysis(symbol: str = "VNM", data_dir: str = "data/historical"):
    """Run comprehensive momentum analysis on a symbol."""
    
    print(f"\n{'='*60}")
    print(f"MOMENTUM STRATEGY ANALYSIS: {symbol}")
    print(f"{'='*60}\n")
    
    # Load data
    df = load_historical_data(symbol, data_dir)
    print(f"Loaded {len(df)} rows of data for {symbol}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}\n")
    
    # Define strategies to test
    strategies = [
        PriceMomentum(short_window=20, long_window=50),
        ROCMomentum(period=12),
        RSIMomentum(period=14),
        MACDMomentum(),
        VolumeWeightedMomentum(),
        OBVMomentum(),
        VPTMomentum(),
        MFIMomentum(),
        TripleMomentum(),
        AcceleratingMomentum()
    ]
    
    # Compare strategies
    comparison = compare_strategies(df, strategies)
    print("Strategy Comparison:")
    print("-" * 100)
    print(comparison.to_string(index=False))
    print()
    
    # Get best strategy
    best_idx = comparison['Sharpe Ratio'].apply(lambda x: float(x)).idxmax()
    best_strategy = strategies[best_idx]
    print(f"\nBest Strategy by Sharpe Ratio: {best_strategy.name}")
    
    # Detailed backtest for best strategy
    result = backtest_strategy(df, best_strategy)
    
    print(f"\nDetailed Results for {best_strategy.name}:")
    print(f"  Total Return: {result['total_return']:.2%}")
    print(f"  Annual Return: {result['annual_return']:.2%}")
    print(f"  Volatility: {result['volatility']:.2%}")
    print(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {result['max_drawdown']:.2%}")
    print(f"  Win Rate: {result['win_rate']:.2%}")
    print(f"  Number of Trades: {int(result['num_trades'])}")
    
    return comparison, result


def run_portfolio_momentum(data_dir: str = "data/historical", top_n: int = 3):
    """Run cross-sectional momentum portfolio analysis."""
    
    print(f"\n{'='*60}")
    print("CROSS-SECTIONAL MOMENTUM PORTFOLIO")
    print(f"{'='*60}\n")
    
    # Load all symbols
    symbols_data = load_all_symbols(data_dir)
    print(f"Loaded data for {len(symbols_data)} symbols: {list(symbols_data.keys())}\n")
    
    # Create portfolio
    portfolio = MomentumPortfolio(lookback=252, holding_period=21, top_n=top_n)
    
    # Get latest date available in all datasets
    min_dates = [df['date'].max() for df in symbols_data.values()]
    latest_date = min(min_dates)
    
    print(f"Portfolio construction date: {latest_date}")
    
    # Calculate current momentum scores
    scores = portfolio.calculate_momentum_scores(symbols_data, latest_date)
    
    print("\nMomentum Scores (12-month return):")
    print("-" * 40)
    for symbol, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {symbol}: {score:.2%}")
    
    # Get recommended weights
    weights = portfolio.get_portfolio_weights(symbols_data, latest_date)
    
    print(f"\nRecommended Portfolio (Top {top_n}):")
    print("-" * 40)
    for symbol, weight in weights.items():
        print(f"  {symbol}: {weight:.2%}")
    
    return scores, weights


if __name__ == "__main__":
    # Run single symbol analysis
    comparison, best_result = run_momentum_analysis("VNM")
    
    # Run portfolio analysis
    scores, weights = run_portfolio_momentum(top_n=3)
    
    # Plot best strategy
    try:
        plot_strategy_performance(best_result)
    except Exception as e:
        print(f"\nNote: Could not plot (matplotlib may not be installed): {e}")
