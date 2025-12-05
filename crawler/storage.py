from pathlib import Path
import pandas as pd


def ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def save_ohlc_csv(symbol: str, df: pd.DataFrame, out_dir: str = "data/historical") -> Path:
    """Save historical OHLC DataFrame to CSV. Overwrites existing file for that symbol.

    Args:
        symbol: stock symbol string used for filename (safe to include exchange prefix)
        df: pandas DataFrame with a Date-like index or a `date` column
        out_dir: directory to store CSV files

    Returns:
        Path to saved CSV file
    """
    out = Path(out_dir) / f"{symbol}_ohlc.csv"
    ensure_dir(out)
    if not df.index.name:
        if "date" in df.columns:
            df = df.set_index("date")
    df.to_csv(out, index=True)
    return out
