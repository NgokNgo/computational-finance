"""Flexible parsers for cafef-like stock pages.

This module provides helpers to extract historical OHLC tables and to parse
current price/fundamental blocks. Many cafef tables use multi-row headers and
merged cells; we prefer `pandas.read_html` (which handles complex headers) and
fall back to a BeautifulSoup-based heuristic when necessary.
"""
from bs4 import BeautifulSoup
import re
import pandas as pd
from datetime import datetime


def _clean_number(text: str):
    if text is None:
        return None
    t = re.sub(r"[\s\u00A0,]+", "", str(text))
    t = t.replace("–", "").replace("%", "")
    try:
        if "." in t and t.count(".") == 1 and "," not in t:
            return float(t)
        # try replacing comma for thousands
        t2 = t.replace(".", "").replace(",", ".")
        return float(t2)
    except Exception:
        return text.strip()


def _is_date_like(s: str) -> bool:
    if s is None:
        return False
    s = str(s).strip()
    # common formats: DD/MM/YYYY, D/M/YYYY, YYYY-MM-DD
    return bool(re.search(r"\b\d{1,4}[/-]\d{1,2}[/-]\d{2,4}\b", s))


def find_first_table_with_date(soup: BeautifulSoup) -> pd.DataFrame:
    """Return the first table-like DataFrame that contains a date-like column.

    Strategy:
    1. Try `pandas.read_html` on the whole HTML — handles multi-row headers.
    2. For each found DataFrame, look for any column that has at least one
       date-like cell and treat that DataFrame as the historical table.
    3. If pandas fails or finds nothing, fall back to a BeautifulSoup-based
       table extraction (previous heuristic).
    """
    html = str(soup)
    # 1) Try pandas.read_html
    try:
        tables = pd.read_html(html, flavor="bs4")
        for df in tables:
            # flatten MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [" | ".join([str(part).strip() for part in col if str(part) and str(part) != 'nan']) for col in df.columns.values]
            # inspect columns for date-like values
            for col in df.columns:
                sample = df[col].dropna().astype(str).head(10).tolist()
                if any(_is_date_like(x) for x in sample):
                    # rename the column to 'date' for convenience if it's not named
                    if str(col).lower() not in ("date", "ngày", "ngay"):
                        df = df.rename(columns={col: "date"})
                    return df
    except Exception:
        # pandas couldn't parse tables — fall back
        pass

    # 2) Fallback: previous BeautifulSoup heuristic
    tables = soup.find_all("table")
    date_regex = re.compile(r"\d{1,4}[/-]\d{1,2}[/-]\d{1,4}")
    for table in tables:
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        rows = []
        for tr in table.find_all("tr"):
            cols = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cols:
                rows.append(cols)
        if not rows:
            continue
        col0 = [r[0] for r in rows if r]
        if any(date_regex.search(c) for c in col0 if c):
            df = pd.DataFrame(rows[1:] if headers else rows)
            if headers and len(headers) == df.shape[1]:
                df.columns = headers
            return df
    return pd.DataFrame()


def parse_stock_page(html: str) -> dict:
    """Parse a stock detail page and return a dict with current price and fundamentals.

    Returns keys like 'symbol', 'timestamp', 'last', 'open', 'high', 'low', 'volume',
    and 'fundamentals' (a dict). Values are heuristically parsed.
    """
    soup = BeautifulSoup(html, "html.parser")
    out = {}

    # Try to find a visible price element (common patterns)
    price_text = None
    candidate = soup.select_one(".boxprice .price, .price, .stock-price, #price")
    if candidate:
        price_text = candidate.get_text(strip=True)
    else:
        # fallback: look for any bold/strong number near known labels
        for strong in soup.find_all(["strong", "b"]):
            txt = strong.get_text(" ", strip=True)
            if re.search(r"\d+[.,]\d+", txt):
                price_text = txt
                break

    if price_text:
        out["last"] = _clean_number(price_text)

    # Look for open/high/low/volume labels nearby
    text = soup.get_text("\n", strip=True)
    labels = {
        "open": ["Giá mở cửa", "Open"],
        "high": ["Cao nhất", "High"],
        "low": ["Thấp nhất", "Low"],
        "volume": ["Khối lượng", "Volume"],
    }
    for key, variants in labels.items():
        for v in variants:
            m = re.search(rf"{re.escape(v)}\s*[:\-]?\s*([\d.,]+)", text, re.IGNORECASE)
            if m:
                out[key] = _clean_number(m.group(1))
                break

    # Attempt to parse simple fundamentals: look for label:value patterns
    fund = {}
    for m in re.finditer(r"([A-Za-z\/%().\u00C0-\u017F ]{2,40})\s*[:\-]\s*([\d.,]+)", text):
        label = m.group(1).strip()
        val = _clean_number(m.group(2))
        fund[label] = val
    if fund:
        out["fundamentals"] = fund

    out["timestamp"] = datetime.utcnow().isoformat()
    return out
