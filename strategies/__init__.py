"""
Strategies Module
=================
Contains all trading strategies organized by category.

Strategy Categories:
- Momentum: Price momentum, Volume-weighted momentum, RSI, MACD, etc.
- Fundamental: Value, Quality, Growth, GARP, Piotroski F-Score, etc.
- Regression: Linear Regression Slope, Channel, Multi-Factor
- TimeSeries: ARIMA, GARCH, ARIMA-GARCH

Usage:
    from strategies import (
        # Momentum
        PriceMomentum,
        RSIMomentum,
        MACDMomentum,
        # Fundamental
        ValueStrategy,
        QualityStrategy,
        GrowthStrategy,
        # Regression
        LinearRegressionSlope,
        LinearRegressionChannel,
        MultiFactorRegression,
        # TimeSeries
        ARIMAStrategy,
        GARCHVolatilityStrategy,
        ARIMAGARCHStrategy,
    )
"""

# Momentum Strategies
from .momentum import (
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
    MomentumPortfolio,
    backtest_strategy,
    compare_strategies,
)

# Fundamental Strategies
from .fundamental import (
    FundamentalMetrics,
    FundamentalStrategy,
    ValueStrategy,
    QualityStrategy,
    GrowthStrategy,
    GARPStrategy,
    PiotroskiStrategy,
    DividendStrategy,
    BalanceSheetStrategy,
    FCFStrategy,
    CompositeStrategy,
    SectorRelativeStrategy,
    FundamentalPortfolio,
    backtest_fundamental_strategy,
)

# Regression Strategies
from .regression import (
    RegressionStrategy,
    LinearRegressionSlope,
    LinearRegressionChannel,
    MultiFactorRegression,
    linear_regression,
    rolling_linear_regression,
    multiple_linear_regression,
    backtest_regression_strategy,
    compare_regression_strategies,
)

# Time-Series Strategies
from .timeseries import (
    TimeSeriesStrategy,
    ARIMAStrategy,
    GARCHVolatilityStrategy,
    ARIMAGARCHStrategy,
    check_stationarity,
    arima_forecast,
    rolling_arima_forecast,
    fit_garch,
    garch_forecast,
    rolling_garch_forecast,
    backtest_timeseries_strategy,
    compare_timeseries_strategies,
)

__all__ = [
    # Momentum
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
    # Fundamental
    'FundamentalMetrics',
    'FundamentalStrategy',
    'ValueStrategy',
    'QualityStrategy',
    'GrowthStrategy',
    'GARPStrategy',
    'PiotroskiStrategy',
    'DividendStrategy',
    'BalanceSheetStrategy',
    'FCFStrategy',
    'CompositeStrategy',
    'SectorRelativeStrategy',
    'FundamentalPortfolio',
    'backtest_fundamental_strategy',
    # Regression
    'RegressionStrategy',
    'LinearRegressionSlope',
    'LinearRegressionChannel',
    'MultiFactorRegression',
    'linear_regression',
    'rolling_linear_regression',
    'multiple_linear_regression',
    'backtest_regression_strategy',
    'compare_regression_strategies',
    # TimeSeries
    'TimeSeriesStrategy',
    'ARIMAStrategy',
    'GARCHVolatilityStrategy',
    'ARIMAGARCHStrategy',
    'check_stationarity',
    'arima_forecast',
    'rolling_arima_forecast',
    'fit_garch',
    'garch_forecast',
    'rolling_garch_forecast',
    'backtest_timeseries_strategy',
    'compare_timeseries_strategies',
]
