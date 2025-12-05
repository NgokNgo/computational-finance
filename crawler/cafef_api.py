"""Direct API client for cafef.vn historical and realtime data.

This module calls cafef's internal JSON APIs directly instead of scraping HTML.
Much faster and more reliable than HTML parsing.
"""
import requests
import pandas as pd
from typing import Optional, List
from datetime import datetime


# API endpoints discovered from cafef.vn
HISTORICAL_API = "https://cafef.vn/du-lieu/Ajax/PageNew/DataHistory/PriceHistory.ashx"
REALTIME_API = "https://cafef.vn/du-lieu/ajax/mobile/smart/ajaxchisothegioi.ashx"

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://cafef.vn/",
    "Accept": "application/json, text/javascript, */*; q=0.01",
}


def fetch_historical_api(
    symbol: str,
    start_date: str = "",
    end_date: str = "",
    page_size: int = 1000,
    max_pages: int = 10,
) -> pd.DataFrame:
    """Fetch historical OHLC data from cafef API.

    Args:
        symbol: Stock symbol (e.g., 'ACV', 'VIC', 'VNM')
        start_date: Start date in DD/MM/YYYY format (empty = no limit)
        end_date: End date in DD/MM/YYYY format (empty = no limit)
        page_size: Number of records per page (max ~1000)
        max_pages: Maximum pages to fetch (to avoid infinite loops)

    Returns:
        DataFrame with columns: date, open, high, low, close, volume, etc.
    """
    all_rows = []
    page_index = 1

    while page_index <= max_pages:
        params = {
            "Symbol": symbol,
            "StartDate": start_date,
            "EndDate": end_date,
            "PageIndex": page_index,
            "PageSize": page_size,
        }
        try:
            resp = requests.get(HISTORICAL_API, params=params, headers=DEFAULT_HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"API error for {symbol} page {page_index}: {e}")
            break

        inner = data.get("Data", {})
        rows = inner.get("Data", [])
        total_count = inner.get("TotalCount", 0)

        if not rows:
            break

        all_rows.extend(rows)

        # Check if we have all data
        if len(all_rows) >= total_count:
            break

        page_index += 1

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Rename columns to standard OHLC names
    column_map = {
        "Ngay": "date",
        "GiaMoCua": "open",
        "GiaCaoNhat": "high",
        "GiaThapNhat": "low",
        "GiaDongCua": "close",
        "GiaDieuChinh": "adj_close",
        "KhoiLuongKhopLenh": "volume",
        "GiaTriKhopLenh": "value",
        "KLThoaThuan": "deal_volume",
        "GtThoaThuan": "deal_value",
        "ThayDoi": "change",
    }
    df = df.rename(columns=column_map)

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
        df = df.sort_values("date")

    return df


def get_available_symbols() -> List[str]:
    """Get list of available symbols from cafef.

    This is a placeholder - cafef doesn't have a public symbols API.
    Returns a default list of popular VN-Index stocks.
    """
    # Common VN-Index constituents
    return [
        "VIC", "VHM", "VCB", "BID", "VNM", "HPG", "MSN", "MWG", "VPB", "ACB",
        "TCB", "FPT", "NVL", "VRE", "SAB", "GAS", "PLX", "POW", "VJC", "HVN",
        "SSI", "VND", "HCM", "MBB", "STB", "EIB", "LPB", "SHB", "TPB", "CTG",
        "ACV", "HDB", "VGC", "DIG", "DXG", "KDH", "NLG", "PDR", "REE", "GMD",
    ]
