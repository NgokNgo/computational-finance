"""Historical OHLC fetcher.

This module fetches a stock's historical data. It first tries cafef's direct
JSON API (fast & reliable), then falls back to HTML scraping if needed.
"""
from typing import Optional
import requests
from bs4 import BeautifulSoup
import pandas as pd
from .cafef_parser import find_first_table_with_date
from .storage import save_ohlc_csv
import re


# Default cafef API endpoint
CAFEF_HISTORICAL_API = "https://cafef.vn/du-lieu/Ajax/PageNew/DataHistory/PriceHistory.ashx"


def fetch_historical_from_api(
    symbol: str,
    start_date: str = "",
    end_date: str = "",
    page_size: int = 1000,
    max_pages: int = 10,
) -> pd.DataFrame:
    """Fetch historical OHLC data directly from cafef API (preferred method).

    Args:
        symbol: Stock symbol (e.g., 'ACV', 'VIC')
        start_date: Start date DD/MM/YYYY (empty = all)
        end_date: End date DD/MM/YYYY (empty = all)
        page_size: Records per page
        max_pages: Max pages to fetch

    Returns:
        DataFrame with OHLC data
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://cafef.vn/",
    }
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
            resp = requests.get(CAFEF_HISTORICAL_API, params=params, headers=headers, timeout=30)
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


def _render_page_with_playwright(url: str, timeout: int = 45) -> str:
    """Render the given URL with Playwright and return the page HTML."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        raise ImportError(
            "Playwright is required to render JS pages. Install with `pip install playwright` "
            "and run `playwright install` (then retry)."
        ) from e
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle", timeout=timeout * 1000)
        page.wait_for_timeout(2000)  # Extra wait for lazy-loaded content
        html = page.content()
        browser.close()
    return html


def fetch_historical(symbol: str, url_template: Optional[str] = None, out_dir: str = "data/historical") -> Optional[str]:
    """Fetch historical data for `symbol` and save to CSV.

    Strategy:
    1. Try cafef JSON API first (fast & reliable)
    2. If API fails and url_template is provided, fall back to HTML scraping
    3. If HTML scraping fails, try Playwright rendering

    Args:
        symbol: Stock symbol
        url_template: Optional URL template for HTML fallback (contains {symbol})
        out_dir: Output directory for CSV

    Returns:
        Path to CSV or None if not found
    """
    df = pd.DataFrame()

    # 1. Try API first (preferred)
    print(f"Trying cafef API for {symbol}...")
    df = fetch_historical_from_api(symbol)

    # 2. Fallback to HTML scraping if API returns empty
    if df.empty and url_template:
        print(f"API returned empty, trying HTML scraping for {symbol}...")
        url = url_template.format(symbol=symbol)
        try:
            resp = requests.get(url, timeout=20, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            resp.raise_for_status()
            html = resp.text
            soup = BeautifulSoup(html, "html.parser")
            df = find_first_table_with_date(soup)

            # 3. Try Playwright if no table in raw HTML
            date_regex = re.compile(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}")
            if df.empty and not date_regex.search(html):
                print(f"No table in raw HTML, trying Playwright for {symbol}...")
                try:
                    rendered = _render_page_with_playwright(url)
                    soup = BeautifulSoup(rendered, "html.parser")
                    df = find_first_table_with_date(soup)
                except ImportError as ie:
                    print(f"Playwright missing: {ie}")
                except Exception as e:
                    print(f"Playwright render failed: {e}")
        except Exception as e:
            print(f"HTML fetch failed for {symbol}: {e}")

    if df.empty:
        return None

    # Ensure date column exists and is properly formatted
    if "date" not in df.columns:
        # Find first column that looks like a date and rename it
        for col in df.columns:
            sample = df[col].astype(str).head(5).tolist()
            if any(re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", s) for s in sample):
                df = df.rename(columns={col: "date"})
                break

    # Convert date and sort
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
            df = df.sort_values("date")
            df = df.set_index("date")
        except Exception:
            pass

    save_ohlc_csv(symbol, df, out_dir=out_dir)
    return f"{out_dir}/{symbol}_ohlc.csv"
