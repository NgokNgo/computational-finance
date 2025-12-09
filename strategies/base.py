import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from abc import ABC, abstractmethod

DEFAULT_INITIAL_CAPITAL = 100000.0
DEFAULT_TRANSACTION_COST = 0.001  # 0.1%
TRADING_DAYS_PER_YEAR = 252


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    All strategy implementations should inherit from this class
    and implement the generate_signals() method.
    
    Attributes:
        name: Strategy identifier name
        signals: Generated trading signals (-1, 0, 1)
        positions: Actual positions after signal shift
    """
    
    def __init__(self, name: str):
        self.name = name
        self.signals: Optional[pd.Series] = None
        self.positions: Optional[pd.Series] = None
        
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement generate_signals()")
        
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate strategy returns based on signals.
        
        Generates signals if not already generated, then calculates:
        - position: Signal shifted by 1 day (enter position next day)
        - market_return: Daily returns of the asset
        - strategy_return: Returns captured by the strategy
        - cumulative_market: Cumulative market returns
        - cumulative_strategy: Cumulative strategy returns
        
        Args:
            df: DataFrame with OHLCV data (must have 'adj_close' column)
            
        Returns:
            DataFrame with additional return columns
        """
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
    
    def __repr__(self) -> str:
        """String representation of strategy."""
        return f"{self.__class__.__name__}(name='{self.name}')"


# =============================================================================
# BACKTEST FUNCTION
# =============================================================================

def backtest_strategy(
    df: pd.DataFrame, 
    strategy: BaseStrategy, 
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    transaction_cost: float = DEFAULT_TRANSACTION_COST
) -> Dict:
    """
    Backtest a trading strategy.
    
    This is the unified backtest function for all strategy types
    (momentum, regression, time-series).
    
    Args:
        df: DataFrame with OHLCV data (must have 'adj_close' column)
        strategy: BaseStrategy instance (any strategy subclass)
        initial_capital: Starting capital (default: 100,000)
        transaction_cost: Transaction cost as percentage (default: 0.1%)
        
    Returns:
        Dictionary with backtest results:
            - strategy: Strategy name
            - total_return: Total return over the period
            - annual_return: Annualized return
            - volatility: Annualized volatility
            - sharpe_ratio: Sharpe ratio (annualized)
            - max_drawdown: Maximum drawdown
            - win_rate: Percentage of winning trades
            - num_trades: Total number of trades
            - final_value: Final portfolio value
            - data: DataFrame with all calculations
    """
    # Generate signals and calculate returns
    result_df = strategy.generate_signals(df)
    result_df = strategy.calculate_returns(result_df)
    
    # Account for transaction costs
    result_df['trade'] = result_df['signal'].diff().abs()
    result_df['tc'] = result_df['trade'] * transaction_cost
    result_df['strategy_return_net'] = result_df['strategy_return'] - result_df['tc']
    result_df['cumulative_strategy_net'] = (1 + result_df['strategy_return_net']).cumprod()
    
    # Calculate metrics
    returns = result_df['strategy_return_net'].dropna()
    n_periods = len(returns)
    
    total_return = result_df['cumulative_strategy_net'].iloc[-1] - 1
    annual_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / n_periods) - 1 if n_periods > 0 else 0
    volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
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


def compare_strategies(
    df: pd.DataFrame, 
    strategies: List[BaseStrategy],
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    transaction_cost: float = DEFAULT_TRANSACTION_COST
) -> pd.DataFrame:
    """
    Compare multiple strategies on the same data.
    
    Args:
        df: DataFrame with OHLCV data
        strategies: List of BaseStrategy instances
        initial_capital: Starting capital
        transaction_cost: Transaction cost as percentage
        
    Returns:
        DataFrame with comparison metrics for each strategy
    """
    results = []
    
    for strategy in strategies:
        result = backtest_strategy(df, strategy, initial_capital, transaction_cost)
        results.append({
            'strategy': result['strategy'],
            'total_return': result['total_return'],
            'annual_return': result['annual_return'],
            'volatility': result['volatility'],
            'sharpe_ratio': result['sharpe_ratio'],
            'max_drawdown': result['max_drawdown'],
            'win_rate': result['win_rate'],
            'num_trades': result['num_trades'],
            'final_value': result['final_value']
        })
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('sharpe_ratio', ascending=False)
    
    return comparison_df
