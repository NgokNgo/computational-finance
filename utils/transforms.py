"""
Data Transformation Module
==========================
Utilities for data resampling, normalization, and quality checking.

Categories:
- Resampling: Weekly, Monthly conversion
- Normalization: Normalize, Standardize, Rank
- Formatting: Percentage, Currency, Number
- Data Quality: Quality checks and validation
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


# =============================================================================
# RESAMPLING
# =============================================================================

def resample_to_weekly(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Resample daily data to weekly.
    
    Args:
        df: DataFrame with daily OHLCV data
        date_col: Name of date column
        
    Returns:
        Weekly resampled DataFrame
    """
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
    """
    Resample daily data to monthly.
    
    Args:
        df: DataFrame with daily OHLCV data
        date_col: Name of date column
        
    Returns:
        Monthly resampled DataFrame
    """
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


def resample_to_period(df: pd.DataFrame, period: str = 'W', 
                        date_col: str = 'date') -> pd.DataFrame:
    """
    Resample daily data to specified period.
    
    Args:
        df: DataFrame with daily OHLCV data
        period: Pandas period string ('W', 'M', 'Q', 'Y')
        date_col: Name of date column
        
    Returns:
        Resampled DataFrame
    """
    df = df.set_index(date_col)
    
    agg_dict = {}
    if 'open' in df.columns:
        agg_dict['open'] = 'first'
    if 'high' in df.columns:
        agg_dict['high'] = 'max'
    if 'low' in df.columns:
        agg_dict['low'] = 'min'
    if 'close' in df.columns:
        agg_dict['close'] = 'last'
    if 'adj_close' in df.columns:
        agg_dict['adj_close'] = 'last'
    if 'volume' in df.columns:
        agg_dict['volume'] = 'sum'
    
    resampled = df.resample(period).agg(agg_dict).dropna()
    return resampled.reset_index()


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_series(series: pd.Series) -> pd.Series:
    """
    Normalize series to 0-1 range (Min-Max scaling).
    
    Args:
        series: Input series
        
    Returns:
        Normalized series (0-1)
    """
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
    return (series - min_val) / (max_val - min_val)


def standardize_series(series: pd.Series) -> pd.Series:
    """
    Standardize series to mean=0, std=1 (Z-score normalization).
    
    Args:
        series: Input series
        
    Returns:
        Standardized series
    """
    mean = series.mean()
    std = series.std()
    if std == 0:
        return pd.Series(0, index=series.index)
    return (series - mean) / std


def rank_percentile(series: pd.Series) -> pd.Series:
    """
    Convert series to percentile ranks (0-1).
    
    Args:
        series: Input series
        
    Returns:
        Percentile ranks
    """
    return series.rank(pct=True)


def rolling_normalize(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling normalization (0-1 within rolling window).
    
    Args:
        series: Input series
        window: Rolling window size
        
    Returns:
        Rolling normalized series
    """
    rolling_min = series.rolling(window=window).min()
    rolling_max = series.rolling(window=window).max()
    
    diff = rolling_max - rolling_min
    diff = diff.replace(0, np.nan)
    
    return (series - rolling_min) / diff


def rolling_standardize(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling standardization (Z-score within rolling window).
    
    Args:
        series: Input series
        window: Rolling window size
        
    Returns:
        Rolling standardized series
    """
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    
    rolling_std = rolling_std.replace(0, np.nan)
    
    return (series - rolling_mean) / rolling_std


def winsorize(series: pd.Series, lower_percentile: float = 0.01, 
              upper_percentile: float = 0.99) -> pd.Series:
    """
    Winsorize series by clipping extreme values.
    
    Args:
        series: Input series
        lower_percentile: Lower percentile to clip
        upper_percentile: Upper percentile to clip
        
    Returns:
        Winsorized series
    """
    lower = series.quantile(lower_percentile)
    upper = series.quantile(upper_percentile)
    return series.clip(lower=lower, upper=upper)


# =============================================================================
# FORMATTING
# =============================================================================

def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage string.
    
    Args:
        value: Decimal value (0.1 = 10%)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value*100:.{decimals}f}%"


def format_currency(value: float, decimals: int = 0, currency: str = "VND") -> str:
    """
    Format value as currency string.
    
    Args:
        value: Numeric value
        decimals: Number of decimal places
        currency: Currency symbol/name
        
    Returns:
        Formatted currency string
    """
    return f"{value:,.{decimals}f} {currency}"


def format_number(value: float, decimals: int = 2) -> str:
    """
    Format number with thousands separator.
    
    Args:
        value: Numeric value
        decimals: Number of decimal places
        
    Returns:
        Formatted number string
    """
    return f"{value:,.{decimals}f}"


def format_ratio(value: float, decimals: int = 2) -> str:
    """
    Format ratio value.
    
    Args:
        value: Ratio value
        decimals: Number of decimal places
        
    Returns:
        Formatted ratio string
    """
    return f"{value:.{decimals}f}"


# =============================================================================
# DATA QUALITY
# =============================================================================

def check_data_quality(df: pd.DataFrame, symbol: str = "") -> Dict[str, Any]:
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
    
    # Check for negative values in price/volume columns
    check_cols = price_cols + ['volume']
    for col in check_cols:
        if col in df.columns:
            negatives = (df[col] < 0).sum()
            if negatives > 0:
                report['negative_values'][col] = negatives
    
    # Check for duplicates
    if 'date' in df.columns:
        report['duplicates'] = df['date'].duplicated().sum()
    
    return report


def print_data_quality_report(report: Dict[str, Any]):
    """
    Print formatted data quality report.
    
    Args:
        report: Dictionary from check_data_quality()
    """
    print(f"\n{'='*50}")
    print(f" Data Quality Report: {report['symbol']}")
    print(f"{'='*50}")
    
    print(f"\n📊 Overview:")
    print(f"   Total Rows:    {report['total_rows']}")
    print(f"   Date Range:    {report['date_range']}")
    
    if report['missing_values']:
        print(f"\n⚠️ Missing Values:")
        for col, count in report['missing_values'].items():
            print(f"   {col}: {count}")
    else:
        print(f"\n✅ No missing values")
    
    if report['zero_values']:
        print(f"\n⚠️ Zero Values in Price Columns:")
        for col, count in report['zero_values'].items():
            print(f"   {col}: {count}")
    
    if report['negative_values']:
        print(f"\n❌ Negative Values:")
        for col, count in report['negative_values'].items():
            print(f"   {col}: {count}")
    
    if report['duplicates'] > 0:
        print(f"\n⚠️ Duplicate Dates: {report['duplicates']}")
    else:
        print(f"\n✅ No duplicate dates")
    
    print(f"\n{'='*50}\n")


def validate_ohlc(df: pd.DataFrame) -> bool:
    """
    Validate OHLC data consistency.
    
    Checks:
    - High >= Open, Close, Low
    - Low <= Open, Close, High
    - All prices > 0
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        True if valid, False otherwise
    """
    if not all(c in df.columns for c in ['open', 'high', 'low', 'close']):
        return False
    
    # Check High is highest
    high_valid = (df['high'] >= df[['open', 'low', 'close']].max(axis=1)).all()
    
    # Check Low is lowest
    low_valid = (df['low'] <= df[['open', 'high', 'close']].min(axis=1)).all()
    
    # Check all positive
    positive_valid = (df[['open', 'high', 'low', 'close']] > 0).all().all()
    
    return high_valid and low_valid and positive_valid


__all__ = [
    # Resampling
    'resample_to_weekly',
    'resample_to_monthly',
    'resample_to_period',
    # Normalization
    'normalize_series',
    'standardize_series',
    'rank_percentile',
    'rolling_normalize',
    'rolling_standardize',
    'winsorize',
    # Formatting
    'format_percentage',
    'format_currency',
    'format_number',
    'format_ratio',
    # Data Quality
    'check_data_quality',
    'print_data_quality_report',
    'validate_ohlc',
]
