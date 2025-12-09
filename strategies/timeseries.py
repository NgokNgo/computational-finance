"""
Time-Series Model-Based Trading Strategies
==========================================
This module implements trading strategies using time-series models:

1. ARIMA Strategy - Forecast prices using ARIMA model
2. GARCH Volatility Strategy - Use volatility forecasting for position sizing
3. Combined ARIMA-GARCH Strategy - Combine return and volatility forecasts

Uses statsmodels for ARIMA and arch for GARCH models.

Author: Computational Finance Project
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import statsmodels for ARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Import arch for GARCH
from arch import arch_model

# Import from utils modules
from utils.data_loader import load_historical_data, load_all_historical as load_all_symbols
from utils.indicators import calculate_returns, calculate_sma
from utils.metrics import calculate_sharpe_ratio, calculate_max_drawdown
from strategies.base import BaseStrategy


# =============================================================================
# ARIMA HELPER FUNCTIONS (USING STATSMODELS)
# =============================================================================

def check_stationarity(series: np.ndarray, significance: float = 0.05) -> Tuple[bool, float]:
    """
    Check stationarity using Augmented Dickey-Fuller test.
    
    Args:
        series: Time series data
        significance: Significance level for the test
        
    Returns:
        Tuple of (is_stationary, p_value)
    """
    try:
        result = adfuller(series, autolag='AIC')
        p_value = result[1]
        is_stationary = p_value < significance
        return is_stationary, p_value
    except Exception:
        return False, 1.0


def arima_forecast(series: np.ndarray, p: int = 1, d: int = 1, q: int = 1, 
                   steps: int = 1) -> Tuple[float, object]:
    """
    Fit ARIMA(p,d,q) using statsmodels and forecast.
    
    Args:
        series: Time series data
        p: AR order
        d: Differencing order
        q: MA order
        steps: Number of steps to forecast
        
    Returns:
        Tuple of (forecast, fitted_model)
    """
    try:
        # Fit ARIMA model using statsmodels
        model = ARIMA(series, order=(p, d, q))
        fitted_model = model.fit()
        
        # Forecast
        forecast = fitted_model.forecast(steps=steps)
        
        # Return single value for steps=1, array otherwise
        if steps == 1:
            return float(forecast.iloc[0]) if hasattr(forecast, 'iloc') else float(forecast[0]), fitted_model
        return forecast, fitted_model
        
    except Exception as e:
        # Fallback to naive forecast (last value)
        return float(series[-1]), None


def rolling_arima_forecast(series: pd.Series, p: int = 1, d: int = 1, q: int = 1,
                            window: int = 100) -> np.ndarray:
    """
    Rolling ARIMA forecast using statsmodels.
    
    Args:
        series: Price series
        p, d, q: ARIMA orders
        window: Training window size
        
    Returns:
        Array of forecasts
    """
    n = len(series)
    forecasts = np.full(n, np.nan)
    
    for i in range(window, n):
        train_data = series.iloc[i-window:i].values
        try:
            forecast, _ = arima_forecast(train_data, p, d, q, steps=1)
            forecasts[i] = forecast
        except Exception:
            forecasts[i] = train_data[-1]  # Naive forecast on failure
    
    return forecasts


# =============================================================================
# GARCH HELPER FUNCTIONS (USING ARCH LIBRARY)
# =============================================================================

def fit_garch(returns: np.ndarray, p: int = 1, q: int = 1) -> Tuple[object, np.ndarray]:
    """
    Fit GARCH(p,q) model using arch library.
    
    GARCH(1,1): σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}
    
    Args:
        returns: Return series (as percentages or decimals)
        p: ARCH order (number of lagged squared returns)
        q: GARCH order (number of lagged variances)
        
    Returns:
        Tuple of (fitted_model, conditional_variance)
    """
    try:
        # Scale returns to percentage for numerical stability
        returns_pct = returns * 100 if np.abs(returns).mean() < 1 else returns
        
        # Create and fit GARCH model
        model = arch_model(returns_pct, vol='Garch', p=p, q=q, 
                          mean='Constant', rescale=False)
        fitted_model = model.fit(disp='off', show_warning=False)
        
        # Get conditional variance (convert back to decimal scale if needed)
        cond_var = fitted_model.conditional_volatility ** 2
        if np.abs(returns).mean() < 1:
            cond_var = cond_var / 10000  # Convert from pct^2 to decimal^2
        
        return fitted_model, cond_var
        
    except Exception as e:
        # Fallback: return simple variance
        var = np.var(returns)
        return None, np.full(len(returns), var)


def garch_forecast(fitted_model: object, returns: np.ndarray, 
                   steps: int = 1) -> np.ndarray:
    """
    Forecast volatility using fitted GARCH model.
    
    Args:
        fitted_model: Fitted arch model
        returns: Return series
        steps: Number of steps to forecast
        
    Returns:
        Array of volatility (std dev) forecasts
    """
    try:
        if fitted_model is None:
            # Fallback to historical volatility
            return np.full(steps, np.std(returns))
        
        # Forecast variance
        forecast = fitted_model.forecast(horizon=steps)
        variance_forecast = forecast.variance.values[-1, :]
        
        # Convert to volatility (std dev) and scale back if needed
        vol_forecast = np.sqrt(variance_forecast)
        if np.abs(returns).mean() < 1:
            vol_forecast = vol_forecast / 100  # Convert from pct to decimal
        
        return vol_forecast
        
    except Exception:
        return np.full(steps, np.std(returns))


def rolling_garch_forecast(returns: pd.Series, p: int = 1, q: int = 1,
                            window: int = 100) -> np.ndarray:
    """
    Rolling GARCH volatility forecast using arch library.
    
    Args:
        returns: Return series
        p, q: GARCH orders
        window: Training window size
        
    Returns:
        Array of volatility forecasts
    """
    n = len(returns)
    vol_forecasts = np.full(n, np.nan)
    
    for i in range(window, n):
        train_data = returns.iloc[i-window:i].values
        try:
            fitted_model, _ = fit_garch(train_data, p, q)
            vol_forecast = garch_forecast(fitted_model, train_data, steps=1)
            vol_forecasts[i] = vol_forecast[0]
        except Exception:
            vol_forecasts[i] = np.std(train_data)  # Fallback to historical vol
    
    return vol_forecasts


# =============================================================================
# STRATEGY 1: ARIMA PRICE FORECAST
# =============================================================================

class ARIMAStrategy(BaseStrategy):
    """
    ARIMA Price Forecast Strategy.
    
    Uses ARIMA model to forecast next-day price.
    - Buy (1) if forecasted price > current price
    - Sell (0) if forecasted price <= current price
    
    Parameters:
        p: AR order (default: 2)
        d: Differencing order (default: 1)
        q: MA order (default: 1)
        window: Training window (default: 100)
    """
    
    def __init__(self, p: int = 2, d: int = 1, q: int = 1, window: int = 100):
        super().__init__("ARIMA Forecast")
        self.p = p
        self.d = d
        self.q = q
        self.window = window
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Rolling ARIMA forecast
        forecasts = rolling_arima_forecast(
            df['adj_close'], 
            p=self.p, d=self.d, q=self.q, 
            window=self.window
        )
        
        df['price_forecast'] = forecasts
        df['forecast_return'] = (df['price_forecast'] - df['adj_close']) / df['adj_close']
        
        # Signal: 1 if forecast > current price (expected increase)
        df['signal'] = np.where(df['price_forecast'] > df['adj_close'], 1, 0)
        
        self.signals = df['signal']
        return df


# =============================================================================
# STRATEGY 2: GARCH VOLATILITY REGIME
# =============================================================================

class GARCHVolatilityStrategy(BaseStrategy):
    """
    GARCH Volatility Regime Strategy.
    
    Uses GARCH to forecast volatility and adjust positions:
    - Full position (1) in low volatility regime
    - Reduced/no position (0) in high volatility regime
    
    The idea is to reduce exposure during turbulent periods.
    
    Parameters:
        p: ARCH order (default: 1)
        q: GARCH order (default: 1)
        window: Training window (default: 100)
        vol_threshold_pct: Percentile for high vol regime (default: 75)
    """
    
    def __init__(self, p: int = 1, q: int = 1, window: int = 100, 
                 vol_threshold_pct: float = 75):
        super().__init__("GARCH Volatility Regime")
        self.p = p
        self.q = q
        self.window = window
        self.vol_threshold_pct = vol_threshold_pct
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Calculate returns
        df['returns'] = df['adj_close'].pct_change()
        
        # Rolling GARCH volatility forecast
        vol_forecasts = rolling_garch_forecast(
            df['returns'].dropna(),
            p=self.p, q=self.q,
            window=self.window
        )
        
        # Align with original dataframe (first row has NaN return)
        df['vol_forecast'] = np.nan
        df.iloc[1:, df.columns.get_loc('vol_forecast')] = vol_forecasts
        
        # Calculate rolling volatility threshold
        df['vol_threshold'] = df['vol_forecast'].rolling(
            window=self.window, min_periods=20
        ).apply(lambda x: np.nanpercentile(x, self.vol_threshold_pct))
        
        # Also calculate historical vol for comparison
        df['hist_vol'] = df['returns'].rolling(20).std()
        
        # Signal: 1 in low vol regime, 0 in high vol regime
        df['vol_regime'] = np.where(df['vol_forecast'] <= df['vol_threshold'], 'low', 'high')
        df['signal'] = np.where(df['vol_forecast'] <= df['vol_threshold'], 1, 0)
        
        # Add momentum filter: only go long if price > SMA
        df['sma_50'] = calculate_sma(df['adj_close'], 50)
        df['signal'] = np.where(
            (df['signal'] == 1) & (df['adj_close'] > df['sma_50']),
            1, 0
        )
        
        self.signals = df['signal']
        return df


# =============================================================================
# STRATEGY 3: COMBINED ARIMA-GARCH
# =============================================================================

class ARIMAGARCHStrategy(BaseStrategy):
    """
    Combined ARIMA-GARCH Strategy.
    
    Combines:
    1. ARIMA for return/price direction forecast
    2. GARCH for volatility-adjusted position sizing
    
    Signal logic:
    - ARIMA predicts positive return AND low volatility -> Full long (1)
    - ARIMA predicts positive return AND high volatility -> Reduced long (0.5)
    - ARIMA predicts negative return -> No position (0)
    
    Parameters:
        arima_p, arima_d, arima_q: ARIMA orders
        garch_p, garch_q: GARCH orders
        window: Training window
    """
    
    def __init__(self, arima_p: int = 2, arima_d: int = 1, arima_q: int = 1,
                 garch_p: int = 1, garch_q: int = 1, window: int = 100):
        super().__init__("ARIMA-GARCH Combined")
        self.arima_p = arima_p
        self.arima_d = arima_d
        self.arima_q = arima_q
        self.garch_p = garch_p
        self.garch_q = garch_q
        self.window = window
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Calculate returns
        df['returns'] = df['adj_close'].pct_change()
        
        # ARIMA price forecast
        price_forecasts = rolling_arima_forecast(
            df['adj_close'],
            p=self.arima_p, d=self.arima_d, q=self.arima_q,
            window=self.window
        )
        df['price_forecast'] = price_forecasts
        df['expected_return'] = (df['price_forecast'] - df['adj_close']) / df['adj_close']
        
        # GARCH volatility forecast
        vol_forecasts = rolling_garch_forecast(
            df['returns'].dropna(),
            p=self.garch_p, q=self.garch_q,
            window=self.window
        )
        df['vol_forecast'] = np.nan
        df.iloc[1:, df.columns.get_loc('vol_forecast')] = vol_forecasts
        
        # Calculate volatility regime
        vol_median = df['vol_forecast'].rolling(window=self.window, min_periods=20).median()
        df['vol_regime'] = np.where(df['vol_forecast'] <= vol_median, 'low', 'high')
        
        # Calculate Sharpe-like ratio: expected return / volatility
        df['sharpe_forecast'] = df['expected_return'] / (df['vol_forecast'] * np.sqrt(252) + 1e-10)
        
        # Combined signal logic
        signals = np.zeros(len(df))
        
        for i in range(len(df)):
            exp_ret = df['expected_return'].iloc[i]
            vol_regime = df['vol_regime'].iloc[i]
            sharpe_fc = df['sharpe_forecast'].iloc[i]
            
            if pd.isna(exp_ret) or pd.isna(vol_regime):
                signals[i] = 0
            elif exp_ret > 0 and sharpe_fc > 0:
                # Positive expected return with favorable risk-adjusted return
                if vol_regime == 'low':
                    signals[i] = 1  # Full position in low vol
                else:
                    signals[i] = 0.5  # Reduced position in high vol
            else:
                signals[i] = 0  # No position
        
        df['signal'] = signals
        self.signals = df['signal']
        return df
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns with position sizing."""
        if self.signals is None:
            self.generate_signals(df)
            
        df = df.copy()
        df['signal'] = self.signals
        df['position'] = df['signal'].shift(1)  # Enter position next day
        df['market_return'] = df['adj_close'].pct_change()
        
        # Strategy return considers position size
        df['strategy_return'] = df['position'] * df['market_return']
        df['cumulative_market'] = (1 + df['market_return']).cumprod()
        df['cumulative_strategy'] = (1 + df['strategy_return']).cumprod()
        
        return df


__all__ = [
    # ARIMA helpers (statsmodels)
    'check_stationarity',
    'arima_forecast',
    'rolling_arima_forecast',
    # GARCH helpers (arch library)
    'fit_garch',
    'garch_forecast',
    'rolling_garch_forecast',
    # Strategies
    'ARIMAStrategy',
    'GARCHVolatilityStrategy',
    'ARIMAGARCHStrategy'
]
