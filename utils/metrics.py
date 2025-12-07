"""
Performance Metrics Module
==========================
Functions for calculating risk-adjusted returns, risk metrics, and performance analysis.

Categories:
- Risk-Adjusted Returns: Sharpe, Sortino, Calmar
- Risk Metrics: Max Drawdown, Volatility
- Return Metrics: CAGR, Win Rate, Profit Factor
- Comprehensive: Full performance summary
"""

import pandas as pd
import numpy as np
from typing import Dict


# =============================================================================
# RISK-ADJUSTED RETURN METRICS
# =============================================================================

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02, 
                           periods_per_year: int = 252) -> float:
    """
    Calculate annualized Sharpe Ratio.
    
    Args:
        returns: Daily returns series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading days per year
        
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    if returns.std() == 0:
        return 0
    return np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02,
                            periods_per_year: int = 252) -> float:
    """
    Calculate annualized Sortino Ratio (uses downside deviation).
    
    Args:
        returns: Daily returns series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading days per year
        
    Returns:
        Sortino ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0
    
    downside_std = downside_returns.std()
    return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std


def calculate_calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate Calmar Ratio (Annual Return / Max Drawdown).
    
    Args:
        returns: Daily returns series
        periods_per_year: Trading days per year
        
    Returns:
        Calmar ratio
    """
    cumulative = (1 + returns).cumprod()
    max_dd = calculate_max_drawdown(cumulative)
    annual_return = (1 + returns.mean()) ** periods_per_year - 1
    
    if max_dd == 0:
        return 0
    return annual_return / abs(max_dd)


def calculate_information_ratio(returns: pd.Series, benchmark_returns: pd.Series,
                                 periods_per_year: int = 252) -> float:
    """
    Calculate Information Ratio (Active Return / Tracking Error).
    
    Args:
        returns: Strategy returns series
        benchmark_returns: Benchmark returns series
        periods_per_year: Trading days per year
        
    Returns:
        Information ratio
    """
    active_returns = returns - benchmark_returns
    tracking_error = active_returns.std()
    
    if tracking_error == 0:
        return 0
    
    return np.sqrt(periods_per_year) * active_returns.mean() / tracking_error


def calculate_treynor_ratio(returns: pd.Series, benchmark_returns: pd.Series,
                            risk_free_rate: float = 0.02, 
                            periods_per_year: int = 252) -> float:
    """
    Calculate Treynor Ratio (Excess Return / Beta).
    
    Args:
        returns: Strategy returns series
        benchmark_returns: Benchmark returns series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading days per year
        
    Returns:
        Treynor ratio
    """
    beta = calculate_beta(returns, benchmark_returns)
    
    if beta == 0:
        return 0
    
    excess_return = returns.mean() * periods_per_year - risk_free_rate
    return excess_return / beta


# =============================================================================
# RISK METRICS
# =============================================================================

def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        cumulative_returns: Cumulative returns series (1 = starting value)
        
    Returns:
        Maximum drawdown (negative value)
    """
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    return drawdown.min()


def calculate_drawdown_series(cumulative_returns: pd.Series) -> pd.Series:
    """
    Calculate drawdown series.
    
    Args:
        cumulative_returns: Cumulative returns series
        
    Returns:
        Drawdown series
    """
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    return drawdown


def calculate_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate annualized volatility."""
    return returns.std() * np.sqrt(periods_per_year)


def calculate_downside_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate annualized downside volatility."""
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        return 0
    return downside_returns.std() * np.sqrt(periods_per_year)


def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR).
    
    Args:
        returns: Daily returns series
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        VaR (negative value representing potential loss)
    """
    return returns.quantile(1 - confidence_level)


def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (CVaR / Expected Shortfall).
    
    Args:
        returns: Daily returns series
        confidence_level: Confidence level
        
    Returns:
        CVaR (average loss beyond VaR)
    """
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()


def calculate_beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calculate Beta (systematic risk).
    
    Args:
        returns: Strategy returns series
        benchmark_returns: Benchmark returns series
        
    Returns:
        Beta coefficient
    """
    covariance = returns.cov(benchmark_returns)
    benchmark_variance = benchmark_returns.var()
    
    if benchmark_variance == 0:
        return 0
    
    return covariance / benchmark_variance


def calculate_alpha(returns: pd.Series, benchmark_returns: pd.Series,
                    risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    """
    Calculate Jensen's Alpha.
    
    Args:
        returns: Strategy returns series
        benchmark_returns: Benchmark returns series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading days per year
        
    Returns:
        Alpha (annualized)
    """
    beta = calculate_beta(returns, benchmark_returns)
    rf_daily = risk_free_rate / periods_per_year
    
    strategy_excess = returns.mean() - rf_daily
    benchmark_excess = benchmark_returns.mean() - rf_daily
    
    alpha_daily = strategy_excess - beta * benchmark_excess
    return alpha_daily * periods_per_year


# =============================================================================
# RETURN METRICS
# =============================================================================

def calculate_cagr(cumulative_returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate Compound Annual Growth Rate.
    
    Args:
        cumulative_returns: Cumulative returns series
        periods_per_year: Trading days per year
        
    Returns:
        CAGR as decimal
    """
    total_periods = len(cumulative_returns)
    if total_periods == 0 or cumulative_returns.iloc[0] == 0:
        return 0
    
    total_return = cumulative_returns.iloc[-1] / cumulative_returns.iloc[0]
    years = total_periods / periods_per_year
    
    if years == 0:
        return 0
    
    return total_return ** (1 / years) - 1


def calculate_total_return(returns: pd.Series) -> float:
    """Calculate total return."""
    return (1 + returns).prod() - 1


def calculate_win_rate(returns: pd.Series) -> float:
    """Calculate win rate (percentage of positive returns)."""
    if len(returns) == 0:
        return 0
    return (returns > 0).sum() / len(returns)


def calculate_profit_factor(returns: pd.Series) -> float:
    """Calculate profit factor (gross profit / gross loss)."""
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0
    return gross_profit / gross_loss


def calculate_avg_win_loss_ratio(returns: pd.Series) -> float:
    """Calculate average win / average loss ratio."""
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    if len(wins) == 0 or len(losses) == 0:
        return 0
    
    avg_win = wins.mean()
    avg_loss = abs(losses.mean())
    
    if avg_loss == 0:
        return float('inf') if avg_win > 0 else 0
    
    return avg_win / avg_loss


def calculate_expectancy(returns: pd.Series) -> float:
    """
    Calculate expectancy (expected return per trade).
    
    Expectancy = (Win Rate × Avg Win) - (Loss Rate × Avg Loss)
    """
    win_rate = calculate_win_rate(returns)
    loss_rate = 1 - win_rate
    
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
    
    return (win_rate * avg_win) - (loss_rate * avg_loss)


# =============================================================================
# COMPREHENSIVE METRICS
# =============================================================================

def calculate_performance_metrics(returns: pd.Series, 
                                   risk_free_rate: float = 0.02,
                                   benchmark_returns: pd.Series = None) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        returns: Daily returns series
        risk_free_rate: Annual risk-free rate
        benchmark_returns: Optional benchmark returns for relative metrics
        
    Returns:
        Dictionary of performance metrics
    """
    cumulative = (1 + returns).cumprod()
    
    metrics = {
        # Returns
        'total_return': calculate_total_return(returns),
        'cagr': calculate_cagr(cumulative),
        
        # Risk
        'volatility': calculate_volatility(returns),
        'downside_volatility': calculate_downside_volatility(returns),
        'max_drawdown': calculate_max_drawdown(cumulative),
        'var_95': calculate_var(returns, 0.95),
        'cvar_95': calculate_cvar(returns, 0.95),
        
        # Risk-Adjusted
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate),
        'sortino_ratio': calculate_sortino_ratio(returns, risk_free_rate),
        'calmar_ratio': calculate_calmar_ratio(returns),
        
        # Trade Statistics
        'win_rate': calculate_win_rate(returns),
        'profit_factor': calculate_profit_factor(returns),
        'avg_win_loss_ratio': calculate_avg_win_loss_ratio(returns),
        'expectancy': calculate_expectancy(returns),
        
        # Additional
        'num_trades': len(returns),
        'num_winning_trades': (returns > 0).sum(),
        'num_losing_trades': (returns < 0).sum(),
    }
    
    # Add benchmark-relative metrics if benchmark provided
    if benchmark_returns is not None:
        metrics.update({
            'alpha': calculate_alpha(returns, benchmark_returns, risk_free_rate),
            'beta': calculate_beta(returns, benchmark_returns),
            'information_ratio': calculate_information_ratio(returns, benchmark_returns),
            'treynor_ratio': calculate_treynor_ratio(returns, benchmark_returns, risk_free_rate),
        })
    
    return metrics


def print_performance_summary(metrics: Dict[str, float], title: str = "Performance Summary"):
    """
    Print formatted performance summary.
    
    Args:
        metrics: Dictionary from calculate_performance_metrics()
        title: Title for the summary
    """
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")
    
    print(f"\n📈 Returns:")
    print(f"   Total Return:     {metrics.get('total_return', 0)*100:>10.2f}%")
    print(f"   CAGR:             {metrics.get('cagr', 0)*100:>10.2f}%")
    
    print(f"\n📊 Risk:")
    print(f"   Volatility:       {metrics.get('volatility', 0)*100:>10.2f}%")
    print(f"   Max Drawdown:     {metrics.get('max_drawdown', 0)*100:>10.2f}%")
    print(f"   VaR (95%):        {metrics.get('var_95', 0)*100:>10.2f}%")
    
    print(f"\n⚖️ Risk-Adjusted:")
    print(f"   Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):>10.2f}")
    print(f"   Sortino Ratio:    {metrics.get('sortino_ratio', 0):>10.2f}")
    print(f"   Calmar Ratio:     {metrics.get('calmar_ratio', 0):>10.2f}")
    
    print(f"\n🎯 Trade Statistics:")
    print(f"   Win Rate:         {metrics.get('win_rate', 0)*100:>10.2f}%")
    print(f"   Profit Factor:    {metrics.get('profit_factor', 0):>10.2f}")
    print(f"   Expectancy:       {metrics.get('expectancy', 0)*100:>10.4f}%")
    
    if 'alpha' in metrics:
        print(f"\n📐 Relative Metrics:")
        print(f"   Alpha:            {metrics.get('alpha', 0)*100:>10.2f}%")
        print(f"   Beta:             {metrics.get('beta', 0):>10.2f}")
        print(f"   Info Ratio:       {metrics.get('information_ratio', 0):>10.2f}")
    
    print(f"\n{'='*50}\n")


__all__ = [
    # Risk-Adjusted Returns
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_calmar_ratio',
    'calculate_information_ratio',
    'calculate_treynor_ratio',
    # Risk Metrics
    'calculate_max_drawdown',
    'calculate_drawdown_series',
    'calculate_volatility',
    'calculate_downside_volatility',
    'calculate_var',
    'calculate_cvar',
    'calculate_beta',
    'calculate_alpha',
    # Return Metrics
    'calculate_cagr',
    'calculate_total_return',
    'calculate_win_rate',
    'calculate_profit_factor',
    'calculate_avg_win_loss_ratio',
    'calculate_expectancy',
    # Comprehensive
    'calculate_performance_metrics',
    'print_performance_summary',
]
