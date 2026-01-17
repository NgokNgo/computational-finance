"""Module phân tích Risk Factors và Asset Allocation.

Bao gồm:
- Volatility analysis
- Maximum Drawdown
- Beta calculation (so với VN-Index)
- Correlation analysis
- Risk-adjusted returns (Sharpe, Sortino)
- Mean-Variance Optimization
- Risk Parity allocation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_returns(df: pd.DataFrame, price_col: str = 'adj_close') -> pd.Series:
    """Tính daily returns."""
    return df[price_col].pct_change()


def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
    """Tính volatility (standard deviation of returns).
    
    Args:
        returns: Series of returns
        annualize: If True, annualize volatility (multiply by sqrt(252))
    """
    vol = returns.std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def calculate_max_drawdown(df: pd.DataFrame, price_col: str = 'adj_close') -> float:
    """Tính Maximum Drawdown."""
    prices = df[price_col]
    cumulative = (1 + prices.pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def calculate_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """Tính Beta (hệ số tương quan với thị trường).
    
    Beta > 1: Stock biến động nhiều hơn thị trường
    Beta < 1: Stock biến động ít hơn thị trường
    Beta = 1: Stock biến động như thị trường
    """
    # Remove NaN values
    valid_idx = stock_returns.notna() & market_returns.notna()
    stock_ret = stock_returns[valid_idx]
    market_ret = market_returns[valid_idx]
    
    if len(stock_ret) < 2:
        return np.nan
    
    covariance = np.cov(stock_ret, market_ret)[0, 1]
    market_variance = np.var(market_ret)
    
    if market_variance == 0:
        return np.nan
    
    return covariance / market_variance


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Tính Sharpe Ratio (risk-adjusted return).
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default 2%)
    """
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    if returns.std() == 0:
        return 0
    return np.sqrt(252) * excess_returns.mean() / returns.std()


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Tính Sortino Ratio (chỉ xét downside risk).
    
    Similar to Sharpe but only considers downside volatility.
    """
    excess_returns = returns - risk_free_rate / 252
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0
    
    return np.sqrt(252) * excess_returns.mean() / downside_returns.std()


def calculate_correlation_matrix(all_data: Dict[str, pd.DataFrame], 
                                 price_col: str = 'adj_close') -> pd.DataFrame:
    """Tính correlation matrix giữa các cổ phiếu."""
    returns_dict = {}
    
    for symbol, df in all_data.items():
        returns_dict[symbol] = calculate_returns(df, price_col)
    
    returns_df = pd.DataFrame(returns_dict)
    return returns_df.corr()


def analyze_portfolio_risk(all_data: Dict[str, pd.DataFrame],
                           market_data: pd.DataFrame = None,
                           price_col: str = 'adj_close') -> pd.DataFrame:
    """Phân tích risk factors cho tất cả cổ phiếu trong portfolio.
    
    Returns:
        DataFrame with risk metrics for each stock
    """
    results = []
    
    for symbol, df in all_data.items():
        returns = calculate_returns(df, price_col).dropna()
        
        if len(returns) < 2:
            continue
        
        metrics = {
            'Symbol': symbol,
            'Volatility': calculate_volatility(returns, annualize=True),
            'Max Drawdown': calculate_max_drawdown(df, price_col),
            'Sharpe Ratio': calculate_sharpe_ratio(returns),
            'Sortino Ratio': calculate_sortino_ratio(returns),
            'Total Return': (df[price_col].iloc[-1] / df[price_col].iloc[0] - 1),
            'Avg Daily Return': returns.mean(),
            'Downside Deviation': returns[returns < 0].std() * np.sqrt(252),
        }
        
        # Calculate Beta if market data is provided
        if market_data is not None:
            market_returns = calculate_returns(market_data, price_col).dropna()
            # Align dates
            common_dates = returns.index.intersection(market_returns.index)
            if len(common_dates) > 1:
                metrics['Beta'] = calculate_beta(
                    returns.loc[common_dates], 
                    market_returns.loc[common_dates]
                )
            else:
                metrics['Beta'] = np.nan
        
        results.append(metrics)
    
    return pd.DataFrame(results)


def mean_variance_optimization(returns_df: pd.DataFrame,
                               target_return: float = None,
                               risk_free_rate: float = 0.02) -> Dict:
    """Mean-Variance Optimization (Markowitz Portfolio Theory).
    
    Args:
        returns_df: DataFrame of returns (columns = stocks)
        target_return: Target annual return (if None, maximize Sharpe)
        risk_free_rate: Annual risk-free rate
    
    Returns:
        Dict with optimal weights and portfolio metrics
    """
    # Calculate expected returns and covariance matrix
    mean_returns = returns_df.mean() * 252  # Annualize
    cov_matrix = returns_df.cov() * 252  # Annualize
    
    n_assets = len(mean_returns)
    
    # Objective function: minimize portfolio variance
    def portfolio_variance(weights):
        return weights.T @ cov_matrix @ weights
    
    # Portfolio return
    def portfolio_return(weights):
        return weights.T @ mean_returns
    
    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]  # Sum to 1
    
    if target_return is not None:
        constraints.append({'type': 'eq', 'fun': lambda w: portfolio_return(w) - target_return})
    
    # Bounds: 0 <= weight <= 1 (no short selling)
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Initial guess: equal weights
    initial_weights = np.array([1/n_assets] * n_assets)
    
    # Optimize
    if target_return is None:
        # Maximize Sharpe Ratio
        def negative_sharpe(weights):
            ret = portfolio_return(weights)
            vol = np.sqrt(portfolio_variance(weights))
            if vol == 0:
                return 0
            return -(ret - risk_free_rate) / vol
        
        result = minimize(negative_sharpe, initial_weights, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
    else:
        result = minimize(portfolio_variance, initial_weights,
                         method='SLSQP', bounds=bounds, constraints=constraints)
    
    # Calculate portfolio metrics
    optimal_weights = result.x
    portfolio_ret = portfolio_return(optimal_weights)
    portfolio_vol = np.sqrt(portfolio_variance(optimal_weights))
    sharpe = (portfolio_ret - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
    
    return {
        'weights': dict(zip(returns_df.columns, optimal_weights)),
        'expected_return': portfolio_ret,
        'volatility': portfolio_vol,
        'sharpe_ratio': sharpe,
        'optimization_success': result.success
    }


def risk_parity_allocation(returns_df: pd.DataFrame) -> Dict:
    """Risk Parity Portfolio Allocation.
    
    Allocate weights such that each asset contributes equally to portfolio risk.
    """
    cov_matrix = returns_df.cov() * 252  # Annualize
    n_assets = len(returns_df.columns)
    
    # Objective: minimize difference in risk contribution
    def risk_parity_objective(weights):
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        marginal_contrib = cov_matrix @ weights
        risk_contrib = weights * marginal_contrib / portfolio_vol
        
        # Minimize variance of risk contributions
        return np.var(risk_contrib)
    
    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_weights = np.array([1/n_assets] * n_assets)
    
    result = minimize(risk_parity_objective, initial_weights,
                     method='SLSQP', bounds=bounds, constraints=constraints)
    
    optimal_weights = result.x
    portfolio_vol = np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights)
    
    return {
        'weights': dict(zip(returns_df.columns, optimal_weights)),
        'volatility': portfolio_vol,
        'optimization_success': result.success
    }


def inverse_volatility_allocation(all_data: Dict[str, pd.DataFrame],
                                  price_col: str = 'adj_close') -> Dict:
    """Inverse Volatility Weighting.
    
    Allocate weights inversely proportional to volatility.
    Lower volatility stocks get higher weights.
    """
    volatilities = {}
    
    for symbol, df in all_data.items():
        returns = calculate_returns(df, price_col).dropna()
        volatilities[symbol] = calculate_volatility(returns, annualize=True)
    
    # Calculate inverse volatility weights
    inv_vols = {s: 1/v for s, v in volatilities.items() if v > 0}
    total_inv_vol = sum(inv_vols.values())
    weights = {s: inv_vol/total_inv_vol for s, inv_vol in inv_vols.items()}
    
    return {
        'weights': weights,
        'volatilities': volatilities
    }


def plot_risk_analysis(risk_df: pd.DataFrame, title: str = "Risk Analysis"):
    """Visualize risk metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Sort by Sharpe Ratio
    risk_df = risk_df.sort_values('Sharpe Ratio', ascending=False)
    
    # 1. Sharpe Ratio
    colors = ['green' if x > 0 else 'red' for x in risk_df['Sharpe Ratio']]
    axes[0, 0].barh(risk_df['Symbol'], risk_df['Sharpe Ratio'], color=colors, alpha=0.7)
    axes[0, 0].axvline(x=0, color='black', linewidth=1)
    axes[0, 0].set_title('Sharpe Ratio', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Volatility vs Return
    axes[0, 1].scatter(risk_df['Volatility'], risk_df['Total Return'], 
                      s=150, alpha=0.6, c=risk_df['Sharpe Ratio'], cmap='RdYlGn')
    for i, row in risk_df.iterrows():
        axes[0, 1].annotate(row['Symbol'], 
                          (row['Volatility'], row['Total Return']),
                          fontsize=8, ha='center')
    axes[0, 1].set_xlabel('Volatility (Annualized)')
    axes[0, 1].set_ylabel('Total Return')
    axes[0, 1].set_title('Risk-Return Profile', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Max Drawdown
    axes[1, 0].barh(risk_df['Symbol'], risk_df['Max Drawdown'], 
                   color='red', alpha=0.7)
    axes[1, 0].set_title('Maximum Drawdown', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Max Drawdown')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Beta (if available)
    if 'Beta' in risk_df.columns:
        valid_beta = risk_df.dropna(subset=['Beta'])
        if len(valid_beta) > 0:
            colors = ['green' if x < 1 else 'red' for x in valid_beta['Beta']]
            axes[1, 1].barh(valid_beta['Symbol'], valid_beta['Beta'], 
                          color=colors, alpha=0.7)
            axes[1, 1].axvline(x=1, color='black', linewidth=1, linestyle='--')
            axes[1, 1].set_title('Beta (Market Sensitivity)', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Beta')
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(corr_matrix: pd.DataFrame):
    """Plot correlation heatmap."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', 
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_allocation_comparison(allocations: Dict[str, Dict], 
                               method_names: List[str]):
    """Compare different allocation methods."""
    fig, axes = plt.subplots(1, len(allocations), figsize=(6*len(allocations), 6))
    
    if len(allocations) == 1:
        axes = [axes]
    
    for ax, (method, alloc), method_name in zip(axes, allocations.items(), method_names):
        weights = alloc['weights']
        symbols = list(weights.keys())
        weight_values = list(weights.values())
        
        # Filter out zero weights
        non_zero = [(s, w) for s, w in zip(symbols, weight_values) if w > 0.01]
        if non_zero:
            symbols, weight_values = zip(*non_zero)
        
        ax.pie(weight_values, labels=symbols, autopct='%1.1f%%',
              startangle=90, colors=plt.cm.Set3.colors[:len(symbols)])
        ax.set_title(f'{method_name}\n(Vol={alloc.get("volatility", 0):.2%})',
                    fontsize=12, fontweight='bold')
    
    plt.suptitle('Asset Allocation Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Risk Factors Analysis Module")
    print("="*60)
    print("Available functions:")
    print("  - calculate_volatility()")
    print("  - calculate_max_drawdown()")
    print("  - calculate_beta()")
    print("  - calculate_sharpe_ratio()")
    print("  - calculate_sortino_ratio()")
    print("  - analyze_portfolio_risk()")
    print("  - mean_variance_optimization()")
    print("  - risk_parity_allocation()")
    print("  - inverse_volatility_allocation()")
    print("  - plot_risk_analysis()")
    print("  - plot_correlation_heatmap()")
