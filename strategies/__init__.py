# Base Strategy Class
from .base import (
    BaseStrategy, 
    backtest_strategy,
    compare_strategies
)

# Momentum Strategies
from .momentum import (
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
    MomentumPortfolio
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
    LinearRegressionSlope,
    LinearRegressionChannel,
    linear_regression,
    rolling_linear_regression,
)

# Time-Series Strategies
from .timeseries import (
    ARIMAStrategy,
    GARCHVolatilityStrategy,
    ARIMAGARCHStrategy,
    check_stationarity,
    arima_forecast,
    rolling_arima_forecast,
    fit_garch,
    garch_forecast,
    rolling_garch_forecast
)

__all__ = [
    # Base
    'BaseStrategy',
    'backtest_strategy',
    'compare_strategies',
    # Momentum
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
    'LinearRegressionSlope',
    'LinearRegressionChannel',
    'linear_regression',
    'rolling_linear_regression',
    # TimeSeries
    'ARIMAStrategy',
    'GARCHVolatilityStrategy',
    'ARIMAGARCHStrategy',
    'check_stationarity',
    'arima_forecast',
    'rolling_arima_forecast',
    'fit_garch',
    'garch_forecast',
    'rolling_garch_forecast',
]
