"""
Alpha Model Module
==================
Contains momentum strategies, fundamental strategies, and portfolio construction tools.

Utility Functions (from utils):
- Data Loading: load_historical_data, load_all_historical, load_fundamental_data, etc.
- Technical Indicators: calculate_rsi, calculate_macd, calculate_obv, etc.
- Performance Metrics: calculate_sharpe_ratio, calculate_max_drawdown, etc.

Momentum Strategies:
- PriceMomentum: Moving average crossover strategy
- ROCMomentum: Rate of change momentum
- RSIMomentum: Relative Strength Index strategy
- MACDMomentum: MACD crossover strategy
- VolumeWeightedMomentum: Price momentum with volume confirmation
- OBVMomentum: On-Balance Volume trend strategy
- VPTMomentum: Volume Price Trend strategy
- MFIMomentum: Money Flow Index strategy
- DualMomentum: Absolute + Relative momentum
- TripleMomentum: Multi-timeframe momentum
- AcceleratingMomentum: Momentum of momentum

Fundamental Strategies:
- ValueStrategy: P/E, P/B based value investing
- QualityStrategy: ROE, ROA, margin-based quality investing
- GrowthStrategy: Revenue and earnings growth
- GARPStrategy: Growth at Reasonable Price
- PiotroskiStrategy: Piotroski F-Score
- DividendStrategy: Dividend yield and growth
- FCFStrategy: Free Cash Flow analysis
- CompositeStrategy: Multi-factor combination

Portfolio Tools:
- MomentumPortfolio: Cross-sectional momentum ranking
- FundamentalPortfolio: Fundamental factor ranking

Backtesting:
- backtest_strategy: Single strategy backtest
- compare_strategies: Compare multiple strategies

Usage:
    from alpha_model import PriceMomentum, backtest_strategy, load_historical_data
    
    df = load_historical_data('VNM', 'data/historical')
    strategy = PriceMomentum(short_window=20, long_window=50)
    result = backtest_strategy(df, strategy)
"""

# Import utilities first
from .utils import (
    # Data Loading
    load_historical_data,
    load_all_historical,
    load_fundamental_data,
    load_all_fundamentals,
    merge_fundamental_data,
    
    # Technical Indicators
    calculate_returns,
    calculate_log_returns,
    calculate_sma,
    calculate_ema,
    calculate_std,
    calculate_rsi,
    calculate_macd,
    calculate_obv,
    calculate_vpt,
    calculate_mfi,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_stochastic,
    
    # Performance Metrics
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio,
    calculate_win_rate,
    calculate_profit_factor,
    calculate_cagr,
    calculate_volatility,
    calculate_performance_metrics,
    
    # Data Transformation
    resample_to_weekly,
    resample_to_monthly,
    normalize_series,
    standardize_series,
    rank_percentile,
    
    # Formatting
    format_percentage,
    format_currency,
    format_number,
)

# Alias for backward compatibility
load_all_symbols = load_all_historical

from .momentum_strategies import (
    # Strategy classes
    MomentumStrategy,
    PriceMomentum,
    ROCMomentum,
    RSIMomentum,
    MACDMomentum,
    VolumeWeightedMomentum,
    OBVMomentum,
    VPTMomentum,
    MFIMomentum,
    DualMomentum,
    TripleMomentum,
    AcceleratingMomentum,
    
    # Portfolio
    MomentumPortfolio,
    
    # Backtesting
    backtest_strategy,
    compare_strategies,
    plot_strategy_performance,
    
    # Main functions
    run_momentum_analysis,
    run_portfolio_momentum,
)

from .fundamental_strategies import (
    # Data classes
    FundamentalMetrics,
    
    # Strategy classes
    FundamentalStrategy,
    ValueStrategy,
    QualityStrategy,
    GrowthStrategy,
    GARPStrategy,
    PiotroskiStrategy,
    DividendStrategy,
    FCFStrategy,
    CompositeStrategy,
    BalanceSheetStrength,
    
    # Portfolio
    FundamentalPortfolio,
    
    # Analysis functions
    calculate_piotroski_score,
    generate_stock_report,
)

__all__ = [
    # Utils - Data Loading
    'load_historical_data',
    'load_all_historical',
    'load_all_symbols',  # backward compatibility
    'load_fundamental_data',
    'load_all_fundamentals',
    'merge_fundamental_data',
    
    # Utils - Technical Indicators
    'calculate_returns',
    'calculate_log_returns',
    'calculate_sma',
    'calculate_ema',
    'calculate_std',
    'calculate_rsi',
    'calculate_macd',
    'calculate_obv',
    'calculate_vpt',
    'calculate_mfi',
    'calculate_atr',
    'calculate_bollinger_bands',
    'calculate_stochastic',
    
    # Utils - Performance Metrics
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_max_drawdown',
    'calculate_calmar_ratio',
    'calculate_win_rate',
    'calculate_profit_factor',
    'calculate_cagr',
    'calculate_volatility',
    'calculate_performance_metrics',
    
    # Utils - Data Transformation
    'resample_to_weekly',
    'resample_to_monthly',
    'normalize_series',
    'standardize_series',
    'rank_percentile',
    
    # Utils - Formatting
    'format_percentage',
    'format_currency',
    'format_number',
    
    # Momentum Strategies
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
    'plot_strategy_performance',
    'run_momentum_analysis',
    'run_portfolio_momentum',
    
    # Fundamental Strategies
    'FundamentalMetrics',
    'FundamentalStrategy',
    'ValueStrategy',
    'QualityStrategy',
    'GrowthStrategy',
    'GARPStrategy',
    'PiotroskiStrategy',
    'DividendStrategy',
    'FCFStrategy',
    'CompositeStrategy',
    'BalanceSheetStrength',
    'FundamentalPortfolio',
    'calculate_piotroski_score',
    'generate_stock_report',
]