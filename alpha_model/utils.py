"""
Utility Functions for Alpha Model
=================================
Common functions shared across momentum and fundamental strategies.

Contents:
- Qlib Integration (Data Conversion, Initialization)
- Data Loading (Historical & Fundamental)
- Technical Indicators
- Performance Metrics
- Visualization Helpers

Author: Computational Finance Project
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import struct
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# QLIB INTEGRATION
# =============================================================================

QLIB_DATA_DIR = "~/.qlib/qlib_data/vn_data"

try:
    import qlib
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False


def init_qlib(data_dir: str = QLIB_DATA_DIR):
    """
    Initialize Qlib. Call once at start.
    
    Args:
        data_dir: Directory for Qlib data storage
    """
    if not QLIB_AVAILABLE:
        raise ImportError("pip install pyqlib")
    
    path = Path(data_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    qlib.init(provider_uri=str(path), region="cn")
    print(f"✓ Qlib initialized: {path}")


def convert_csv_to_qlib(csv_dir: str = "../data/historical", output_dir: str = QLIB_DATA_DIR):
    """
    Convert CSV files to Qlib binary format.
    
    Args:
        csv_dir: Directory containing CSV files (*_ohlc.csv)
        output_dir: Output directory for Qlib data
        
    Returns:
        Path to output directory
    """
    csv_path = Path(csv_dir)
    out_path = Path(output_dir).expanduser()
    
    # Create directories
    (out_path / "calendars").mkdir(parents=True, exist_ok=True)
    (out_path / "instruments").mkdir(parents=True, exist_ok=True)
    (out_path / "features").mkdir(parents=True, exist_ok=True)
    
    all_dates = set()
    symbols_info = []
    all_data = {}
    
    # Read all CSV files
    for csv_file in csv_path.glob("*_ohlc.csv"):
        symbol = csv_file.stem.replace("_ohlc", "").lower()
        
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.lower()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        all_dates.update(df['date'].tolist())
        all_data[symbol] = df
        symbols_info.append((symbol, df['date'].min(), df['date'].max()))
        print(f"  Read: {symbol} ({len(df)} rows)")
    
    # Create calendar (sorted dates)
    calendar = sorted(all_dates)
    date_to_idx = {d: i for i, d in enumerate(calendar)}
    
    # Save calendar
    with open(out_path / "calendars" / "day.txt", 'w') as f:
        for d in calendar:
            f.write(d.strftime('%Y-%m-%d') + '\n')
    print(f"  Calendar: {len(calendar)} days")
    
    # Save instruments
    with open(out_path / "instruments" / "all.txt", 'w') as f:
        for sym, start, end in symbols_info:
            f.write(f"{sym}\t{start.strftime('%Y-%m-%d')}\t{end.strftime('%Y-%m-%d')}\n")
    
    # Convert each symbol to binary format
    for symbol, df in all_data.items():
        symbol_dir = out_path / "features" / symbol
        symbol_dir.mkdir(exist_ok=True)
        
        # Map dates to calendar indices
        df['cal_idx'] = df['date'].map(date_to_idx)
        
        # For each feature column
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                continue
                
            # Create full array aligned with calendar
            start_idx = int(df['cal_idx'].iloc[0])
            end_idx = int(df['cal_idx'].iloc[-1])
            
            # Create array with NaN for missing dates
            arr_len = end_idx - start_idx + 1
            values = np.full(arr_len, np.nan, dtype=np.float32)
            
            # Fill in actual values
            for _, row in df.iterrows():
                idx = int(row['cal_idx']) - start_idx
                values[idx] = float(row[col])
            
            # Write Qlib binary format
            bin_file = symbol_dir / f"{col}.day.bin"
            with open(bin_file, 'wb') as f:
                # Header: start_index as float32 (Qlib convention)
                f.write(np.array([start_idx], dtype='<f').tobytes())
                # Data: float32 array
                f.write(values.astype('<f').tobytes())
        
        print(f"  ✓ {symbol}")
    
    print(f"\n✓ Converted {len(symbols_info)} symbols to Qlib format")
    print(f"  Output: {out_path}")
    return out_path


def load_data_with_qlib(symbols: List[str] = None, 
                        start_date: str = None, 
                        end_date: str = None,
                        fields: List[str] = None,
                        csv_dir: str = "data/historical") -> Dict[str, pd.DataFrame]:
    """
    Load data using Qlib with automatic preprocessing.
    Falls back to CSV loading if Qlib is not available.
    
    Args:
        symbols: List of symbols to load (None = all available)
        start_date: Start date string 'YYYY-MM-DD'
        end_date: End date string 'YYYY-MM-DD'
        fields: List of fields to load (default: OHLCV)
        csv_dir: Directory containing CSV files
        
    Returns:
        Dictionary mapping symbol -> preprocessed DataFrame
    """
    if fields is None:
        fields = ['$open', '$high', '$low', '$close', '$volume']
    
    # Try Qlib first
    if QLIB_AVAILABLE:
        try:
            from qlib.data import D
            
            # Initialize Qlib if needed
            try:
                init_qlib()
            except:
                pass
            
            # Get instruments
            if symbols is None:
                instruments = D.instruments(market='all')
                symbols = D.list_instruments(instruments)
            
            data = {}
            for symbol in symbols:
                try:
                    df = D.features(
                        [symbol], 
                        fields, 
                        start_time=start_date, 
                        end_time=end_date,
                        freq='day'
                    )
                    if not df.empty:
                        df = df.reset_index()
                        df.columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
                        df = df.drop('symbol', axis=1)
                        df['adj_close'] = df['close']  # Qlib data is adjusted
                        data[symbol] = df
                except Exception as e:
                    print(f"Qlib error for {symbol}: {e}")
            
            if data:
                print(f"✓ Loaded {len(data)} symbols via Qlib")
                return data
                
        except Exception as e:
            print(f"Qlib not available or error: {e}")
    
    # Fallback to CSV loading with preprocessing
    print("Loading from CSV with preprocessing...")
    return load_all_historical(csv_dir)


# =============================================================================
# DATA LOADING - HISTORICAL
# =============================================================================

def preprocess_historical_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess historical data: handle missing values, outliers, and data quality.
    
    Args:
        df: Raw DataFrame with OHLCV data
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Ensure date column is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    
    # Define numeric columns
    numeric_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume', 'value']
    
    for col in numeric_cols:
        if col in df.columns:
            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle missing values
    # 1. Forward fill for price columns (use previous day's value)
    price_cols = ['open', 'high', 'low', 'close', 'adj_close']
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].ffill()
    
    # 2. Fill volume with 0 if missing (no trading)
    if 'volume' in df.columns:
        df['volume'] = df['volume'].fillna(0)
    
    # 3. Handle outliers using IQR method for prices
    for col in price_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.01)
            Q3 = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=Q1, upper=Q3)
    
    # 4. Ensure OHLC consistency: High >= Open, Close, Low and Low <= Open, Close, High
    if all(c in df.columns for c in ['open', 'high', 'low', 'close']):
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
    
    # 5. Drop rows where critical data is still missing
    df = df.dropna(subset=['adj_close', 'close'])
    
    return df.reset_index(drop=True)


def load_historical_data(symbol: str, data_dir: str = "data/historical", 
                         preprocess: bool = True) -> pd.DataFrame:
    """
    Load historical OHLCV data for a symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'VNM', 'VIC')
        data_dir: Directory containing historical data files
        preprocess: Whether to apply preprocessing (default True)
        
    Returns:
        DataFrame with columns: date, adj_close, close, volume, open, high, low
    """
    filepath = Path(data_dir) / f"{symbol}_ohlc.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Clean numeric columns
    numeric_cols = ['adj_close', 'close', 'volume', 'value', 'open', 'high', 'low']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Apply preprocessing if requested
    if preprocess:
        df = preprocess_historical_data(df)
    
    return df


def load_all_historical(data_dir: str = "data/historical") -> Dict[str, pd.DataFrame]:
    """
    Load historical data for all available symbols.
    
    Args:
        data_dir: Directory containing historical data files
        
    Returns:
        Dictionary mapping symbol -> DataFrame
    """
    data_path = Path(data_dir)
    symbols_data = {}
    
    for file in data_path.glob("*_ohlc.csv"):
        symbol = file.stem.replace("_ohlc", "")
        try:
            symbols_data[symbol] = load_historical_data(symbol, data_dir)
        except Exception as e:
            print(f"Error loading {symbol}: {e}")
    
    return symbols_data


# =============================================================================
# DATA LOADING - FUNDAMENTAL
# =============================================================================

def load_fundamental_data(symbol: str, data_dir: str = "data/fundamental") -> Dict[str, pd.DataFrame]:
    """
    Load all fundamental data for a symbol.
    
    Args:
        symbol: Stock ticker symbol
        data_dir: Directory containing fundamental data
        
    Returns:
        Dictionary with keys: ratios, income, balance, cashflow, overview
    """
    base_path = Path(data_dir) / symbol
    
    data = {}
    files = ['ratios', 'income', 'balance', 'cashflow', 'overview']
    
    for file in files:
        filepath = base_path / f"{file}.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            data[file] = df
    
    return data


def load_all_fundamentals(data_dir: str = "data/fundamental") -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Load fundamental data for all available symbols.
    
    Args:
        data_dir: Directory containing fundamental data
        
    Returns:
        Nested dictionary: symbol -> {ratios, income, balance, cashflow, overview}
    """
    data_path = Path(data_dir)
    all_data = {}
    
    for folder in data_path.iterdir():
        if folder.is_dir():
            symbol = folder.name
            try:
                all_data[symbol] = load_fundamental_data(symbol, data_dir)
            except Exception as e:
                print(f"Error loading {symbol}: {e}")
    
    return all_data


def merge_fundamental_data(fund_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all fundamental dataframes into one.
    
    Args:
        fund_data: Dictionary from load_fundamental_data()
        
    Returns:
        Merged DataFrame with all fundamental metrics
    """
    dfs = []
    
    # Start with ratios as base
    if 'ratios' in fund_data:
        dfs.append(fund_data['ratios'])
    
    # Merge income statement
    if 'income' in fund_data:
        income = fund_data['income'].copy()
        income = income.drop(columns=['ticker'], errors='ignore')
        dfs.append(income)
    
    # Merge balance sheet
    if 'balance' in fund_data:
        balance = fund_data['balance'].copy()
        balance = balance.drop(columns=['ticker'], errors='ignore')
        dfs.append(balance)
    
    # Merge cashflow
    if 'cashflow' in fund_data:
        cashflow = fund_data['cashflow'].copy()
        cashflow = cashflow.drop(columns=['ticker'], errors='ignore')
        dfs.append(cashflow)
    
    if not dfs:
        return pd.DataFrame()
    
    # Merge on quarter and year
    result = dfs[0]
    for df in dfs[1:]:
        result = result.merge(df, on=['quarter', 'year'], how='outer', suffixes=('', '_dup'))
        # Remove duplicate columns
        result = result.loc[:, ~result.columns.str.endswith('_dup')]
    
    return result.sort_values(['year', 'quarter']).reset_index(drop=True)


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate simple returns over specified periods."""
    return prices.pct_change(periods)


def calculate_log_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate log returns over specified periods."""
    return np.log(prices / prices.shift(periods))


def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return series.rolling(window=window).mean()


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def calculate_std(series: pd.Series, window: int) -> pd.Series:
    """Calculate Rolling Standard Deviation."""
    return series.rolling(window=window).std()


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss
    
    Args:
        prices: Price series
        period: Lookback period (default 14)
        
    Returns:
        RSI values (0-100)
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, 
                   signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: Price series
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)
        
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_obv(prices: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).
    OBV accumulates volume based on price direction.
    
    Args:
        prices: Price series
        volume: Volume series
        
    Returns:
        OBV series
    """
    price_change = prices.diff()
    obv = pd.Series(index=prices.index, dtype=float)
    obv.iloc[0] = 0
    
    for i in range(1, len(prices)):
        if price_change.iloc[i] > 0:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif price_change.iloc[i] < 0:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv


def calculate_vpt(prices: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate Volume Price Trend (VPT).
    VPT = Previous VPT + Volume × (Today's Close − Previous Close) / Previous Close
    """
    price_change_pct = prices.pct_change()
    vpt = (volume * price_change_pct).cumsum()
    return vpt


def calculate_mfi(high: pd.Series, low: pd.Series, close: pd.Series, 
                  volume: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Money Flow Index (MFI) - Volume-weighted RSI.
    
    Args:
        high, low, close: Price series
        volume: Volume series
        period: Lookback period
        
    Returns:
        MFI values (0-100)
    """
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    
    tp_diff = typical_price.diff()
    
    positive_flow = pd.Series(0.0, index=close.index)
    negative_flow = pd.Series(0.0, index=close.index)
    
    positive_flow[tp_diff > 0] = raw_money_flow[tp_diff > 0]
    negative_flow[tp_diff < 0] = raw_money_flow[tp_diff < 0]
    
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()
    
    money_ratio = positive_mf / negative_mf
    mfi = 100 - (100 / (1 + money_ratio))
    
    return mfi


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, 
                  period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR) for volatility.
    
    Args:
        high, low, close: Price series
        period: Lookback period
        
    Returns:
        ATR series
    """
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def calculate_bollinger_bands(prices: pd.Series, window: int = 20, 
                               num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        prices: Price series
        window: Moving average window
        num_std: Number of standard deviations
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle = calculate_sma(prices, window)
    std = calculate_std(prices, window)
    
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    
    return upper, middle, lower


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                         k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator (%K and %D).
    
    Args:
        high, low, close: Price series
        k_period: %K lookback period
        d_period: %D smoothing period
        
    Returns:
        Tuple of (%K, %D)
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    
    return k, d


# =============================================================================
# PERFORMANCE METRICS
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
    
    return total_return ** (1 / years) - 1


def calculate_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate annualized volatility."""
    return returns.std() * np.sqrt(periods_per_year)


def calculate_performance_metrics(returns: pd.Series, 
                                   risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        returns: Daily returns series
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Dictionary of performance metrics
    """
    cumulative = (1 + returns).cumprod()
    
    return {
        'total_return': cumulative.iloc[-1] - 1,
        'cagr': calculate_cagr(cumulative),
        'volatility': calculate_volatility(returns),
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate),
        'sortino_ratio': calculate_sortino_ratio(returns, risk_free_rate),
        'max_drawdown': calculate_max_drawdown(cumulative),
        'calmar_ratio': calculate_calmar_ratio(returns),
        'win_rate': calculate_win_rate(returns),
        'profit_factor': calculate_profit_factor(returns)
    }


# =============================================================================
# DATA TRANSFORMATION UTILITIES
# =============================================================================

def resample_to_weekly(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """Resample daily data to weekly."""
    df = df.set_index(date_col)
    weekly = df.resample('W').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'adj_close': 'last',
        'volume': 'sum'
    }).dropna()
    return weekly.reset_index()


def resample_to_monthly(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """Resample daily data to monthly."""
    df = df.set_index(date_col)
    monthly = df.resample('M').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'adj_close': 'last',
        'volume': 'sum'
    }).dropna()
    return monthly.reset_index()


def normalize_series(series: pd.Series) -> pd.Series:
    """Normalize series to 0-1 range."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
    return (series - min_val) / (max_val - min_val)


def standardize_series(series: pd.Series) -> pd.Series:
    """Standardize series to mean=0, std=1."""
    mean = series.mean()
    std = series.std()
    if std == 0:
        return pd.Series(0, index=series.index)
    return (series - mean) / std


def rank_percentile(series: pd.Series) -> pd.Series:
    """Convert series to percentile ranks (0-1)."""
    return series.rank(pct=True)


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage string."""
    return f"{value:.{decimals}%}"


def format_currency(value: float, decimals: int = 0) -> str:
    """Format value as currency string (VND)."""
    return f"{value:,.{decimals}f} VND"


def format_number(value: float, decimals: int = 2) -> str:
    """Format number with thousands separator."""
    return f"{value:,.{decimals}f}"


# =============================================================================
# EXPORTS
# =============================================================================

def check_data_quality(df: pd.DataFrame, symbol: str = "") -> Dict[str, any]:
    """
    Check data quality and return statistics.
    
    Args:
        df: DataFrame to check
        symbol: Symbol name for reporting
        
    Returns:
        Dictionary with quality metrics
    """
    report = {
        'symbol': symbol,
        'total_rows': len(df),
        'date_range': f"{df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else 'N/A',
        'missing_values': {},
        'zero_values': {},
        'negative_values': {},
        'duplicates': 0
    }
    
    # Check for missing values
    for col in df.columns:
        missing = df[col].isna().sum()
        if missing > 0:
            report['missing_values'][col] = missing
    
    # Check for zero values in price columns
    price_cols = ['open', 'high', 'low', 'close', 'adj_close']
    for col in price_cols:
        if col in df.columns:
            zeros = (df[col] == 0).sum()
            if zeros > 0:
                report['zero_values'][col] = zeros
    
    # Check for negative values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        negatives = (df[col] < 0).sum()
        if negatives > 0:
            report['negative_values'][col] = negatives
    
    # Check for duplicates
    if 'date' in df.columns:
        report['duplicates'] = df['date'].duplicated().sum()
    
    return report


def print_data_quality_report(all_data: Dict[str, pd.DataFrame]):
    """Print data quality report for all symbols."""
    print("\n" + "="*70)
    print("DATA QUALITY REPORT")
    print("="*70)
    
    for symbol, df in all_data.items():
        report = check_data_quality(df, symbol)
        
        issues = []
        if report['missing_values']:
            issues.append(f"Missing: {report['missing_values']}")
        if report['zero_values']:
            issues.append(f"Zeros: {report['zero_values']}")
        if report['duplicates'] > 0:
            issues.append(f"Duplicates: {report['duplicates']}")
        
        status = "✓" if not issues else "⚠"
        print(f"\n{status} {symbol}: {report['total_rows']} rows, {report['date_range']}")
        
        for issue in issues:
            print(f"   {issue}")
    
    print("\n" + "="*70)


__all__ = [
    # Qlib Integration
    'QLIB_DATA_DIR',
    'QLIB_AVAILABLE',
    'init_qlib',
    'convert_csv_to_qlib',
    'load_data_with_qlib',
    
    # Data Loading & Preprocessing
    'preprocess_historical_data',
    'load_historical_data',
    'load_all_historical',
    'load_fundamental_data',
    'load_all_fundamentals',
    'merge_fundamental_data',
    'check_data_quality',
    'print_data_quality_report',
    
    # Technical Indicators
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
    
    # Performance Metrics
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_max_drawdown',
    'calculate_calmar_ratio',
    'calculate_win_rate',
    'calculate_profit_factor',
    'calculate_cagr',
    'calculate_volatility',
    'calculate_performance_metrics',
    
    # Data Transformation
    'resample_to_weekly',
    'resample_to_monthly',
    'normalize_series',
    'standardize_series',
    'rank_percentile',
    
    # Formatting
    'format_percentage',
    'format_currency',
    'format_number',
]
