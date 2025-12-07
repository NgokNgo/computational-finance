"""
Linear Regression-Based Trading Strategies
==========================================
This module implements trading strategies using linear regression models:

1. Linear Regression Slope Strategy - Trend direction based on regression slope
2. Linear Regression Channel Strategy - Mean reversion within regression channels
3. Multi-Factor Regression Strategy - Predict returns using multiple features

Author: Computational Finance Project
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import sklearn for linear regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Import from utils modules
from utils.data_loader import load_historical_data, load_all_historical as load_all_symbols
from utils.indicators import calculate_returns, calculate_sma, calculate_rsi
from utils.metrics import calculate_sharpe_ratio, calculate_max_drawdown


# =============================================================================
# LINEAR REGRESSION HELPER FUNCTIONS
# =============================================================================

def linear_regression(y: np.ndarray) -> Tuple[float, float, float]:
    """
    Simple linear regression using sklearn's LinearRegression.
    
    Args:
        y: Target values (prices)
        
    Returns:
        Tuple of (slope, intercept, r_squared)
    """
    n = len(y)
    if n < 2:
        return 0, 0, 0
    
    # Prepare features (time index)
    X = np.arange(n).reshape(-1, 1)
    
    # Fit sklearn LinearRegression
    model = LinearRegression()
    model.fit(X, y)
    
    slope = model.coef_[0]
    intercept = model.intercept_
    
    # Calculate R-squared using sklearn
    y_pred = model.predict(X)
    r_squared = r2_score(y, y_pred)
    
    return slope, intercept, r_squared


def rolling_linear_regression(series: pd.Series, window: int) -> pd.DataFrame:
    """
    Calculate rolling linear regression statistics using sklearn.
    
    Args:
        series: Price series
        window: Rolling window size
        
    Returns:
        DataFrame with slope, intercept, r_squared, upper_band, lower_band
    """
    n = len(series)
    slopes = np.full(n, np.nan)
    intercepts = np.full(n, np.nan)
    r_squares = np.full(n, np.nan)
    reg_values = np.full(n, np.nan)
    upper_bands = np.full(n, np.nan)
    lower_bands = np.full(n, np.nan)
    
    # Prepare X for sklearn (time index)
    X_window = np.arange(window).reshape(-1, 1)
    
    for i in range(window - 1, n):
        y = series.iloc[i - window + 1:i + 1].values
        
        # Fit sklearn LinearRegression
        model = LinearRegression()
        model.fit(X_window, y)
        
        slopes[i] = model.coef_[0]
        intercepts[i] = model.intercept_
        
        # Get predictions and R²
        y_pred = model.predict(X_window)
        r_squares[i] = r2_score(y, y_pred)
        
        # Current regression value (end of line)
        reg_values[i] = model.predict([[window - 1]])[0]
        
        # Calculate standard error for bands
        std_err = np.std(y - y_pred)
        
        upper_bands[i] = reg_values[i] + 2 * std_err
        lower_bands[i] = reg_values[i] - 2 * std_err
    
    return pd.DataFrame({
        'slope': slopes,
        'intercept': intercepts,
        'r_squared': r_squares,
        'reg_value': reg_values,
        'upper_band': upper_bands,
        'lower_band': lower_bands
    }, index=series.index)


def multiple_linear_regression(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Multiple linear regression using sklearn's LinearRegression.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target values
        
    Returns:
        Tuple of (coefficients with intercept as first element, r_squared)
    """
    try:
        # Fit sklearn LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        
        # Combine intercept and coefficients [intercept, coef1, coef2, ...]
        coefficients = np.concatenate([[model.intercept_], model.coef_])
        
        # Calculate R-squared using sklearn
        y_pred = model.predict(X)
        r_squared = r2_score(y, y_pred)
        
        return coefficients, r_squared
    except Exception:
        return np.zeros(X.shape[1] + 1), 0


# =============================================================================
# REGRESSION STRATEGY BASE CLASS
# =============================================================================

class RegressionStrategy:
    """Base class for regression-based strategies."""
    
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


# =============================================================================
# STRATEGY 1: LINEAR REGRESSION SLOPE
# =============================================================================

class LinearRegressionSlope(RegressionStrategy):
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


# =============================================================================
# STRATEGY 2: LINEAR REGRESSION CHANNEL
# =============================================================================

class LinearRegressionChannel(RegressionStrategy):
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
            std_err = np.std(y - y_pred)
            
            upper_bands[i] = reg_val + self.num_std * std_err
            lower_bands[i] = reg_val - self.num_std * std_err
        
        df['lr_channel_mid'] = reg_values
        df['lr_channel_upper'] = upper_bands
        df['lr_channel_lower'] = lower_bands
        
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
        
        df['signal'] = signals
        self.signals = df['signal']
        return df


# =============================================================================
# STRATEGY 3: MULTI-FACTOR REGRESSION
# =============================================================================

class MultiFactorRegression(RegressionStrategy):
    """
    Multi-Factor Regression Strategy.
    
    Uses multiple technical indicators as features to predict next-day returns.
    Generates signals based on predicted return direction.
    
    Features used:
    - Momentum (past N-day returns)
    - RSI
    - Volume ratio
    - Volatility
    - Price distance from SMA
    
    Parameters:
        lookback: Window for calculating features (default: 20)
        train_window: Rolling training window (default: 252)
        retrain_freq: Frequency to retrain model (default: 21)
    """
    
    def __init__(self, lookback: int = 20, train_window: int = 252, retrain_freq: int = 21):
        super().__init__("Multi-Factor Regression")
        self.lookback = lookback
        self.train_window = train_window
        self.retrain_freq = retrain_freq
        
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create feature matrix from price data."""
        features = pd.DataFrame(index=df.index)
        
        # Feature 1: Momentum (past returns)
        features['momentum_5'] = df['adj_close'].pct_change(5)
        features['momentum_10'] = df['adj_close'].pct_change(10)
        features['momentum_20'] = df['adj_close'].pct_change(20)
        
        # Feature 2: RSI
        features['rsi'] = calculate_rsi(df['adj_close'], 14) / 100  # Normalize to 0-1
        
        # Feature 3: Volume ratio
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Feature 4: Volatility (rolling std of returns)
        returns = df['adj_close'].pct_change()
        features['volatility'] = returns.rolling(20).std()
        
        # Feature 5: Price distance from SMA (normalized)
        sma_50 = calculate_sma(df['adj_close'], 50)
        features['sma_distance'] = (df['adj_close'] - sma_50) / sma_50
        
        # Feature 6: Trend strength (linear regression R²)
        reg_stats = rolling_linear_regression(df['adj_close'], 20)
        features['trend_strength'] = reg_stats['r_squared']
        
        # Target: Next day return
        features['target'] = df['adj_close'].pct_change().shift(-1)
        
        return features
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Create features
        features = self._create_features(df)
        feature_cols = ['momentum_5', 'momentum_10', 'momentum_20', 
                        'rsi', 'volume_ratio', 'volatility', 
                        'sma_distance', 'trend_strength']
        
        # Initialize predictions
        n = len(df)
        predictions = np.full(n, np.nan)
        coefficients = np.full((n, len(feature_cols) + 1), np.nan)
        
        # Rolling prediction with sklearn
        min_train = self.train_window
        last_train = 0
        current_model = None
        current_scaler = None
        
        for i in range(min_train, n - 1):
            # Retrain model periodically
            if current_model is None or (i - last_train) >= self.retrain_freq:
                # Training data
                train_start = max(0, i - self.train_window)
                train_data = features.iloc[train_start:i].dropna()
                
                if len(train_data) < 50:  # Minimum samples
                    continue
                    
                X_train = train_data[feature_cols].values
                y_train = train_data['target'].values
                
                # Scale features for better regression performance
                current_scaler = StandardScaler()
                X_train_scaled = current_scaler.fit_transform(X_train)
                
                # Fit sklearn LinearRegression
                current_model = LinearRegression()
                current_model.fit(X_train_scaled, y_train)
                last_train = i
            
            # Make prediction for next day
            if current_model is not None and current_scaler is not None:
                X_pred = features.iloc[i][feature_cols].values
                if not np.any(np.isnan(X_pred)):
                    X_pred_scaled = current_scaler.transform(X_pred.reshape(1, -1))
                    predictions[i] = current_model.predict(X_pred_scaled)[0]
                    # Store coefficients (intercept first, then scaled coefficients)
                    coefficients[i] = np.concatenate([[current_model.intercept_], current_model.coef_])
        
        df['predicted_return'] = predictions
        
        # Store feature values
        for col in feature_cols:
            df[col] = features[col]
        
        # Signal: 1 if predicted return is positive
        df['signal'] = np.where(df['predicted_return'] > 0, 1, 0)
        
        self.signals = df['signal']
        self.coefficients = coefficients
        return df


# =============================================================================
# BACKTESTING FOR REGRESSION STRATEGIES
# =============================================================================

def backtest_regression_strategy(df: pd.DataFrame, strategy: RegressionStrategy, 
                                  initial_capital: float = 100000,
                                  transaction_cost: float = 0.001) -> Dict:
    """
    Backtest a regression-based strategy.
    
    Args:
        df: DataFrame with OHLCV data
        strategy: RegressionStrategy instance
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
    n_periods = len(returns)
    annual_return = (1 + total_return) ** (252 / n_periods) - 1 if n_periods > 0 else 0
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


def compare_regression_strategies(df: pd.DataFrame, 
                                   strategies: List[RegressionStrategy]) -> pd.DataFrame:
    """Compare multiple regression strategies on the same data."""
    results = []
    
    for strategy in strategies:
        result = backtest_regression_strategy(df, strategy)
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


__all__ = [
    # Helper functions
    'linear_regression',
    'rolling_linear_regression',
    'multiple_linear_regression',
    # Strategies
    'RegressionStrategy',
    'LinearRegressionSlope',
    'LinearRegressionChannel',
    'MultiFactorRegression',
    # Backtesting
    'backtest_regression_strategy',
    'compare_regression_strategies',
]
