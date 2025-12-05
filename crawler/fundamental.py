"""Fundamental data fetcher for Vietnamese stocks.

This module fetches financial ratios, overview, and other fundamental data
from TCBS API (public, no auth required).
"""
import requests
import pandas as pd
from typing import Optional, List, Dict
from pathlib import Path


# TCBS API endpoints (public)
TCBS_BASE = "https://apipubaws.tcbs.com.vn/tcanalysis/v1"

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}


def fetch_overview(symbol: str) -> Dict:
    """Fetch company overview info.

    Returns dict with keys like:
    - exchange, shortName, industry, industryEn
    - noEmployees, noShareholders, foreignPercent
    - stockRating, deltaInWeek, deltaInMonth, deltaInYear
    - outstandingShare, issueShare
    """
    url = f"{TCBS_BASE}/ticker/{symbol}/overview"
    try:
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Error fetching overview for {symbol}: {e}")
        return {}


def fetch_financial_ratios(symbol: str, yearly: bool = True, all_data: bool = True) -> List[Dict]:
    """Fetch financial ratios (P/E, P/B, ROE, ROA, EPS, etc.)

    Returns list of dicts, each containing:
    - ticker, quarter, year
    - priceToEarning (P/E), priceToBook (P/B)
    - roe, roa, earningPerShare (EPS), bookValuePerShare
    - dividend, grossProfitMargin, operatingMargin, netProfitMargin
    - daysReceivable, daysInventory, daysPayable
    - currentPayment, quickPayment, equityOnTotalAsset
    """
    yearly_param = "1" if yearly else "0"
    all_param = "true" if all_data else "false"
    url = f"{TCBS_BASE}/finance/{symbol}/financialratio?yearly={yearly_param}&isAll={all_param}"
    try:
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Error fetching financial ratios for {symbol}: {e}")
        return []


def fetch_income_statement(symbol: str, yearly: bool = True) -> List[Dict]:
    """Fetch income statement data (revenue, profit, etc.)

    Returns list of dicts with quarterly/yearly income data.
    """
    yearly_param = "1" if yearly else "0"
    url = f"{TCBS_BASE}/finance/{symbol}/incomestatement?yearly={yearly_param}&isAll=true"
    try:
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Error fetching income statement for {symbol}: {e}")
        return []


def fetch_balance_sheet(symbol: str, yearly: bool = True) -> List[Dict]:
    """Fetch balance sheet data (assets, liabilities, equity)."""
    yearly_param = "1" if yearly else "0"
    url = f"{TCBS_BASE}/finance/{symbol}/balancesheet?yearly={yearly_param}&isAll=true"
    try:
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Error fetching balance sheet for {symbol}: {e}")
        return []


def fetch_cash_flow(symbol: str, yearly: bool = True) -> List[Dict]:
    """Fetch cash flow statement data."""
    yearly_param = "1" if yearly else "0"
    url = f"{TCBS_BASE}/finance/{symbol}/cashflow?yearly={yearly_param}&isAll=true"
    try:
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Error fetching cash flow for {symbol}: {e}")
        return []


def fetch_all_fundamental(symbol: str) -> Dict:
    """Fetch all fundamental data for a symbol.

    Returns dict with keys: overview, ratios, income, balance, cashflow
    """
    return {
        "overview": fetch_overview(symbol),
        "ratios": fetch_financial_ratios(symbol),
        "income": fetch_income_statement(symbol),
        "balance": fetch_balance_sheet(symbol),
        "cashflow": fetch_cash_flow(symbol),
    }


def save_fundamental_csv(symbol: str, out_dir: str = "data/fundamental") -> Dict[str, str]:
    """Fetch and save all fundamental data to CSV files.

    Creates per-symbol folder structure:
    - {out_dir}/{symbol}/overview.csv
    - {out_dir}/{symbol}/ratios.csv
    - {out_dir}/{symbol}/income.csv
    - {out_dir}/{symbol}/balance.csv
    - {out_dir}/{symbol}/cashflow.csv

    Returns dict mapping data type to file path.
    """
    # Create per-symbol subfolder
    symbol_dir = Path(out_dir) / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    data = fetch_all_fundamental(symbol)

    # Overview - single row
    if data["overview"]:
        df = pd.DataFrame([data["overview"]])
        path = symbol_dir / "overview.csv"
        df.to_csv(path, index=False)
        paths["overview"] = str(path)

    # Ratios - time series
    if data["ratios"]:
        df = pd.DataFrame(data["ratios"])
        # Sort by year and quarter
        if "year" in df.columns:
            df = df.sort_values(["year", "quarter"] if "quarter" in df.columns else ["year"])
        path = symbol_dir / "ratios.csv"
        df.to_csv(path, index=False)
        paths["ratios"] = str(path)

    # Income statement
    if data["income"]:
        df = pd.DataFrame(data["income"])
        if "year" in df.columns:
            df = df.sort_values(["year", "quarter"] if "quarter" in df.columns else ["year"])
        path = symbol_dir / "income.csv"
        df.to_csv(path, index=False)
        paths["income"] = str(path)

    # Balance sheet
    if data["balance"]:
        df = pd.DataFrame(data["balance"])
        if "year" in df.columns:
            df = df.sort_values(["year", "quarter"] if "quarter" in df.columns else ["year"])
        path = symbol_dir / "balance.csv"
        df.to_csv(path, index=False)
        paths["balance"] = str(path)

    # Cash flow
    if data["cashflow"]:
        df = pd.DataFrame(data["cashflow"])
        if "year" in df.columns:
            df = df.sort_values(["year", "quarter"] if "quarter" in df.columns else ["year"])
        path = symbol_dir / "cashflow.csv"
        df.to_csv(path, index=False)
        paths["cashflow"] = str(path)

    return paths


def get_latest_ratios(symbol: str) -> Dict:
    """Get the most recent financial ratios for a symbol.

    Returns dict with keys like P/E, P/B, ROE, ROA, EPS, etc.
    """
    ratios = fetch_financial_ratios(symbol)
    if not ratios:
        return {}
    
    # Get most recent (first item is usually latest)
    latest = ratios[0]
    
    # Map to friendly names
    return {
        "symbol": symbol,
        "year": latest.get("year"),
        "quarter": latest.get("quarter"),
        "P/E": latest.get("priceToEarning"),
        "P/B": latest.get("priceToBook"),
        "ROE": latest.get("roe"),
        "ROA": latest.get("roa"),
        "EPS": latest.get("earningPerShare"),
        "BVPS": latest.get("bookValuePerShare"),
        "dividend": latest.get("dividend"),
        "gross_margin": latest.get("grossProfitMargin"),
        "operating_margin": latest.get("operatingMargin"),
        "net_margin": latest.get("netProfitMargin"),
        "current_ratio": latest.get("currentPayment"),
        "quick_ratio": latest.get("quickPayment"),
        "debt_to_equity": latest.get("equityOnLiability"),
    }
