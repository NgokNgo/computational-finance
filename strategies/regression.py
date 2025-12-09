"""
Linear Regression-Based Trading Strategies
==========================================
This module implements trading strategies using linear regression models:

1. Linear Regression Slope Strategy - Trend direction based on regression slope
2. Linear Regression Channel Strategy - Mean reversion within regression channels
3. Multi-Factor Regression Strategy - Predict returns using multiple features

"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler

from strategies.base import BaseStrategy


# =============================================================================
# LINEAR REGRESSION HELPER FUNCTIONS
# =============================================================================

def linear_regression(y: np.ndarray):
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 2:
        return 0.0, 0.0, 0.0, np.full(n, np.nan)

    x = np.arange(n)
    # Use polyfit for a simple linear fit
    try:
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    except Exception:
        slope, intercept, r_squared = 0.0, float(y[-1]) if n > 0 else 0.0, 0.0
        y_pred = np.full(n, intercept)

    return float(slope), float(intercept), float(r_squared), y_pred


def rolling_linear_regression(series: pd.Series, window: int) -> pd.DataFrame:
    # preserve original index so returned DataFrame aligns with input
    series = series.astype(float)  # ...existing index preserved (no reset_index)
    n = len(series)
    slopes = np.full(n, np.nan)
    intercepts = np.full(n, np.nan)
    r_squares = np.full(n, np.nan)
    reg_values = np.full(n, np.nan)
    upper_bands = np.full(n, np.nan)
    lower_bands = np.full(n, np.nan)

    if window < 2:
        raise ValueError("window must be >= 2")

    for i in range(window - 1, n):
        y = series.iloc[i - window + 1:i + 1].values
        try:
            slope, intercept, r2, y_pred = linear_regression(y)
            slopes[i] = slope
            intercepts[i] = intercept
            r_squares[i] = r2

            # regression predicted value at last point of the window
            reg_values[i] = float(y_pred[-1])

            # standard error of residuals (sample std)
            resid = y - y_pred
            std_err = np.std(resid, ddof=1) if resid.size > 1 else 0.0

            upper_bands[i] = reg_values[i] + 2.0 * std_err
            lower_bands[i] = reg_values[i] - 2.0 * std_err
        except Exception:
            # leave NaNs for problematic windows
            continue

    # return DataFrame with the original series index so assignments align by label
    return pd.DataFrame({
        'slope': slopes,
        'intercept': intercepts,
        'r_squared': r_squares,
        'reg_value': reg_values,
        'upper_band': upper_bands,
        'lower_band': lower_bands
    }, index=series.index)


# =============================================================================
# STRATEGIES 
# =============================================================================

class LinearRegressionSlope(BaseStrategy):
    """
    Linear Regression Slope Strategy.
    
    Uses the slope of a rolling linear regression to determine trend direction.
    - Buy (1) when slope is positive and significant (R² above threshold)
    - Sell (0) when slope is negative or R² below threshold
    
    Parameters:
        window: Rolling regression window (default: 20)
        r2_threshold: Minimum R² for signal confidence (default: 0.5)
    """
    
    def __init__(self, window: int = 20, r2_threshold: float = 0.5):
        super().__init__("Linear Regression Slope")
        self.window = window
        self.r2_threshold = r2_threshold
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Calculate rolling regression
        reg_stats = rolling_linear_regression(df['adj_close'], self.window)
        
        df['lr_slope'] = reg_stats['slope']
        df['lr_r_squared'] = reg_stats['r_squared']
        df['lr_value'] = reg_stats['reg_value']
        
        # Normalize slope for better interpretation (as % of price)
        df['lr_slope_pct'] = df['lr_slope'] / df['adj_close'] * 100
        
        # Signal: positive slope AND sufficient R²
        df['signal'] = np.where(
            (df['lr_slope'] > 0) & (df['lr_r_squared'] >= self.r2_threshold),
            1, 0
        )
        
        self.signals = df['signal']
        return df

class LinearRegressionChannel(BaseStrategy):
    """
    Linear Regression Channel Strategy (Mean Reversion).
    
    Uses regression bands to identify overbought/oversold conditions:
    - Buy when price touches lower band (oversold)
    - Sell when price touches upper band (overbought)
    
    Parameters:
        window: Rolling regression window (default: 50)
        num_std: Number of standard deviations for bands (default: 2)
    """
    
    def __init__(self, window: int = 50, num_std: float = 2.0):
        super().__init__("Linear Regression Channel")
        self.window = window
        self.num_std = num_std
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Calculate rolling regression with custom bands
        n = len(df)
        reg_values = np.full(n, np.nan)
        upper_bands = np.full(n, np.nan)
        lower_bands = np.full(n, np.nan)
        
        for i in range(self.window - 1, n):
            y = df['adj_close'].iloc[i - self.window + 1:i + 1].values
            slope, intercept, _ = linear_regression(y)
            
            # Current regression value
            reg_val = slope * (self.window - 1) + intercept
            reg_values[i] = reg_val
            
            # Calculate bands
            x = np.arange(self.window)
            y_pred = slope * x + intercept
            std_err = np.std(y - y_pred, ddof=1) if y.size > 1 else 0.0
            
            upper_bands[i] = reg_val + self.num_std * std_err
            lower_bands[i] = reg_val - self.num_std * std_err
        
        # convert numpy arrays to Series with DataFrame index to ensure alignment
        df['lr_channel_mid'] = pd.Series(reg_values, index=df.index)
        df['lr_channel_upper'] = pd.Series(upper_bands, index=df.index)
        df['lr_channel_lower'] = pd.Series(lower_bands, index=df.index)
        
        # Calculate position within channel (0 = lower band, 1 = upper band)
        channel_width = df['lr_channel_upper'] - df['lr_channel_lower']
        df['lr_channel_position'] = np.where(
            channel_width > 0,
            (df['adj_close'] - df['lr_channel_lower']) / channel_width,
            0.5
        )
        
        # Mean reversion signals with state machine
        df['signal'] = 0
        position = 0
        signals = []
        
        for i in range(len(df)):
            if pd.isna(df['lr_channel_lower'].iloc[i]):
                signals.append(0)
                continue
                
            price = df['adj_close'].iloc[i]
            lower = df['lr_channel_lower'].iloc[i]
            upper = df['lr_channel_upper'].iloc[i]
            mid = df['lr_channel_mid'].iloc[i]
            
            # Buy when price below lower band
            if price <= lower and position == 0:
                position = 1
            # Sell when price above upper band or crosses mid from below
            elif price >= upper and position == 1:
                position = 0
                
            signals.append(position)
        
        # assign signals as a Series to preserve index alignment
        df['signal'] = pd.Series(signals, index=df.index)
        self.signals = df['signal']
        return df
    

__all__ = [
    # Helper functions
    'linear_regression',
    'rolling_linear_regression',
    # Strategies
    'LinearRegressionSlope',
    'LinearRegressionChannel'
]
