"""
Utils Module - Common Utilities
===============================
Shared utilities for data loading, technical indicators, and performance metrics.

Submodules:
- data_loader: Load historical and fundamental data
- indicators: Technical indicators (RSI, MACD, OBV, etc.)
- metrics: Performance metrics (Sharpe, Sortino, Max Drawdown, etc.)
- transforms: Data transformation utilities

Usage:
    from utils import load_historical_data, calculate_rsi
    from utils.metrics import calculate_sharpe_ratio
"""

from .data_loader import (
    # Historical Data
    load_historical_data,
    load_all_historical,
    # Fundamental Data
    load_fundamental_data,
    load_all_fundamentals,
    merge_fundamental_data,
    # Qlib Integration
    init_qlib,
    convert_csv_to_qlib,
    load_data_with_qlib,
    QLIB_AVAILABLE,
)

from .indicators import (
    # Basic
    calculate_returns,
    calculate_log_returns,
    calculate_sma,
    calculate_ema,
    calculate_std,
    # Momentum Indicators
    calculate_rsi,
    calculate_macd,
    calculate_stochastic,
    # Volume Indicators
    calculate_obv,
    calculate_vpt,
    calculate_mfi,
    # Volatility Indicators
    calculate_atr,
    calculate_bollinger_bands,
)

from .metrics import (
    # Risk-Adjusted Returns
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    # Risk Metrics
    calculate_max_drawdown,
    calculate_volatility,
    # Return Metrics
    calculate_cagr,
    calculate_win_rate,
    calculate_profit_factor,
    # Comprehensive
    calculate_performance_metrics,
)

from .transforms import (
    # Resampling
    resample_to_weekly,
    resample_to_monthly,
    # Normalization
    normalize_series,
    standardize_series,
    rank_percentile,
    # Formatting
    format_percentage,
    format_currency,
    format_number,
    # Data Quality
    check_data_quality,
    print_data_quality_report,
)

# Legacy optimization module (use utils.optimizer instead)
try:
    from .optimizer import (
        # Data splitting
        train_val_test_split,
        split_by_date,
        print_split_info,
        # Objective functions
        sharpe_objective,
        calmar_objective,
        sortino_objective,
        combined_objective,
        # Optimization with Validation
        optimize_onestock,
        optimize_universal,
        walk_forward_optimization,
        # Robustness
        bootstrap_performance,
    )
except ImportError:
    # optimize.py removed - use utils.optimizer instead
    pass

from .visualize import (
    plot_strategy_analysis,
    print_backtest_results,
    plot_comparison_charts,
)

from .risk_factors import (
    # Risk Metrics
    calculate_volatility as calc_volatility_rf,
    calculate_max_drawdown as calc_max_drawdown_rf,
    calculate_beta,
    calculate_sharpe_ratio as calc_sharpe_rf,
    calculate_sortino_ratio as calc_sortino_rf,
    calculate_correlation_matrix,
    # Portfolio Analysis
    analyze_portfolio_risk,
    # Allocation Methods
    mean_variance_optimization,
    risk_parity_allocation,
    inverse_volatility_allocation,
    # Visualization
    plot_risk_analysis,
    plot_correlation_heatmap,
    plot_allocation_comparison,
)

# Lean-inspired Modular Optimizer (from optimizer submodule)
try:
    from .optimizer import (
        # Enums
        OptimizationStrategy,
        ObjectiveDirection,
        ConstraintType,
        # Result Models
        OptimizationResult,
        OptimizationSummary,
        OptimizerConfig,
        # Constraints
        Constraint,
        TradeCountConstraint,
        SharpeConstraint,
        DrawdownConstraint,
        ReturnConstraint,
        # Builders
        ObjectiveBuilder,
        # Base Classes
        OptimizerBase,
        GridSearchOptimizer,
        BayesianOptimizer,
        OptunaOptimizer,
        # Alternative Strategies
        RandomSearchOptimizer,
        GeneticAlgorithmOptimizer,
        OptimizerFactory,
        # Analysis
        ParameterSensitivity,
        ResultsAnalyzer,
        RobustnessTester,
        # High-level API
        Optimizer,
        quick_optimize,
    )
    
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False

__all__ = [
    # Data Loading
    'load_historical_data',
    'load_all_historical',
    'load_fundamental_data',
    'load_all_fundamentals',
    'merge_fundamental_data',
    'init_qlib',
    'convert_csv_to_qlib',
    'load_data_with_qlib',
    'QLIB_AVAILABLE',
    # Indicators
    'calculate_returns',
    'calculate_log_returns',
    'calculate_sma',
    'calculate_ema',
    'calculate_std',
    'calculate_rsi',
    'calculate_macd',
    'calculate_stochastic',
    'calculate_obv',
    'calculate_vpt',
    'calculate_mfi',
    'calculate_atr',
    'calculate_bollinger_bands',
    # Metrics
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_calmar_ratio',
    'calculate_max_drawdown',
    'calculate_volatility',
    'calculate_cagr',
    'calculate_win_rate',
    'calculate_profit_factor',
    'calculate_performance_metrics',
    # Transforms
    'resample_to_weekly',
    'resample_to_monthly',
    'normalize_series',
    'standardize_series',
    'rank_percentile',
    'format_percentage',
    'format_currency',
    'format_number',
    'check_data_quality',
    'print_data_quality_report',
    # Optimization with Validation
    'train_val_test_split',
    'split_by_date',
    'print_split_info',
    'sharpe_objective',
    'calmar_objective',
    'sortino_objective',
    'combined_objective',
    'optimize_onestock',
    'optimize_universal',
    'walk_forward_optimization',
    'parameter_sensitivity_analysis',
    'bootstrap_performance',
    # Visualization
    'plot_strategy_analysis',
    'print_backtest_results',
    'plot_comparison_charts',
    # Risk Factors & Asset Allocation
    'calc_volatility_rf',
    'calc_max_drawdown_rf',
    'calculate_beta',
    'calc_sharpe_rf',
    'calc_sortino_rf',
    'calculate_correlation_matrix',
    'analyze_portfolio_risk',
    'mean_variance_optimization',
    'risk_parity_allocation',
    'inverse_volatility_allocation',
    'plot_risk_analysis',
    'plot_correlation_heatmap',
    'plot_allocation_comparison',
    # Lean-inspired Optimizer
    'OptimizationStrategy',
    'ObjectiveDirection',
    'ConstraintType',
    'OptimizationResult',
    'OptimizationSummary',
    'OptimizerConfig',
    'Constraint',
    'TradeCountConstraint',
    'SharpeConstraint',
    'DrawdownConstraint',
    'ReturnConstraint',
    'ObjectiveBuilder',
    'OptimizerBase',
    'GridSearchOptimizer',
    'BayesianOptimizer',
    'OptunaOptimizer',
    'RandomSearchOptimizer',
    'GeneticAlgorithmOptimizer',
    'OptimizerFactory',
    'ParameterSensitivity',
    'ResultsAnalyzer',
    'RobustnessTester',
    'Optimizer',
    'quick_optimize',
    'OPTIMIZER_AVAILABLE',
]
