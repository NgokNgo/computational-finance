"""
Data Loading Module
===================
Functions for loading historical and fundamental data.

Features:
- Historical OHLCV data loading with preprocessing
- Fundamental data loading (ratios, income, balance, cashflow)
- Qlib integration for advanced data handling
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
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
    print(f"✓ Qlib initialized")


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
# HISTORICAL DATA LOADING
# =============================================================================

def load_historical_data(symbol: str, data_dir: str = "data/historical") -> pd.DataFrame:
    """
    Load historical OHLCV data for a symbol (no preprocessing).
    
    Args:
        symbol: Stock ticker symbol (e.g., 'VNM', 'VIC')
        data_dir: Directory containing historical data files
        
    Returns:
        DataFrame with columns: date, open, high, low, close, volume, adj_close
    """
    filepath = Path(data_dir) / f"{symbol}_ohlc.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df.columns = df.columns.str.lower()
    
    # Create adj_close if not present
    if 'adj_close' not in df.columns:
        df['adj_close'] = df['close']
    
    return df


def load_all_historical(data_dir: str = "data/historical") -> tuple:
    """
    Load historical data for all available symbols (no preprocessing).
    
    Args:
        data_dir: Directory containing historical data files
        
    Returns:
        Tuple of (symbols_data dict, summary_df DataFrame)
    """
    data_path = Path(data_dir)
    symbols_data = {}
    
    for csv_file in sorted(data_path.glob("*_ohlc.csv")):
        symbol = csv_file.stem.replace("_ohlc", "")
        try:
            df = pd.read_csv(csv_file)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            df.columns = df.columns.str.lower()
            
            # Create adj_close if not present
            if 'adj_close' not in df.columns:
                df['adj_close'] = df['close']
            
            symbols_data[symbol] = df
        except Exception as e:
            print(f"Error loading {symbol}: {e}")
    
    print(f"✓ Loaded {len(symbols_data)} stocks from {data_dir}")
    print(f"Symbols: {', '.join(sorted(symbols_data.keys())[:10])}...")
    
    # Compute summary statistics
    summary_stats = []
    for symbol, df in symbols_data.items():
        # Skip if not enough data or all NaN
        if df['adj_close'].isna().all() or len(df.dropna(subset=['adj_close'])) < 2:
            print(f"Skipping {symbol}: insufficient data")
            continue
        
        # Remove NaN values for calculation
        df_clean = df.dropna(subset=['adj_close'])
        
        total_return = (df_clean['adj_close'].iloc[-1] / df_clean['adj_close'].iloc[0]) - 1
        volatility = df_clean['adj_close'].pct_change().std() * np.sqrt(252)  # Annualized volatility
        summary_stats.append({
            'symbol': symbol,
            'total_return': total_return,
            'volatility': volatility,
            'start_date': df_clean['date'].iloc[0],
            'end_date': df_clean['date'].iloc[-1],
            'num_days': len(df_clean)
        })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df = summary_df.sort_values(by='num_days', ascending=False).reset_index(drop=True)
    print("✓ Computed summary statistics for all stocks\n")
    print(summary_df)
    
    return symbols_data, summary_df


# =============================================================================
# FUNDAMENTAL DATA LOADING
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


__all__ = [
    # Qlib
    'init_qlib',
    'convert_csv_to_qlib',
    'load_data_with_qlib',
    'QLIB_AVAILABLE',
    # Historical
    'load_historical_data',
    'load_all_historical',
    # Fundamental
    'load_fundamental_data',
    'load_all_fundamentals',
    'merge_fundamental_data',
]
