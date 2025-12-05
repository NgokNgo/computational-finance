"""Utilities to load VN-Index symbols from cafef or a local file.

The web structure can change across sites; we provide a generic scraper that
looks for uppercase symbol-like tokens in tables. For reliable runs, pass a
prepared `symbols.txt` file where each line is a symbol.
"""
from typing import List
import requests
from bs4 import BeautifulSoup
import re


def load_symbols_from_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def fetch_symbols_from_cafef(url: str) -> List[str]:
    """Fetch VN-Index or exchange component page and extract uppercase tokens.

    This is heuristic — update `url` to the page listing components.
    """
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    tokens = set()
    # find table cells and links commonly holding symbols
    for tag in soup.find_all(["td", "a", "span", "li"]):
        txt = tag.get_text(" ", strip=True)
        if not txt:
            continue
        # common VN ticker pattern: uppercase letters 2-5 length, optionally with . or -
        for m in re.finditer(r"\b[A-Z]{2,5}\b", txt):
            tokens.add(m.group(0))
    return sorted(tokens)
