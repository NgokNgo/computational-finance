#!/usr/bin/env python3
"""CLI entrypoint for the VN-Index crawler.

Usage examples are in the project README. This script supports subcommands:
- `symbols` to fetch or load symbol lists
- `historical` to fetch historical OHLC for one or more symbols (uses cafef API by default)
- `fundamental` to fetch fundamental data (P/E, ROE, EPS, etc.) from TCBS API
- `realtime` to poll symbols and append realtime rows

"""
import argparse
from crawler import symbols as symbols_mod
from crawler.historical import fetch_historical
from crawler.fundamental import save_fundamental_csv, get_latest_ratios
import sys


def cmd_symbols(args):
    if args.from_file:
        syms = symbols_mod.load_symbols_from_file(args.from_file)
        print("Loaded", len(syms), "symbols from file")
        for s in syms:
            print(s)
    elif args.from_url:
        syms = symbols_mod.fetch_symbols_from_cafef(args.from_url)
        print("Fetched", len(syms), "symbols from URL")
        for s in syms:
            print(s)
    else:
        print("Provide --from-file PATH or --from-url URL")


def cmd_historical(args):
    if not args.symbol and not args.symbols_file:
        print("Provide --symbol SYMBOL or --symbols-file FILE")
        sys.exit(1)
    syms = []
    if args.symbol:
        syms.append(args.symbol)
    if args.symbols_file:
        syms.extend(symbols_mod.load_symbols_from_file(args.symbols_file))
    for s in syms:
        try:
            # url_template is now optional (API is used by default)
            path = fetch_historical(s, url_template=args.url_template, out_dir=args.outdir)
            if path:
                print(f"Saved historical for {s} -> {path}")
            else:
                print(f"No historical data found for {s}")
        except Exception as e:
            print(f"Error fetching historical for {s}: {e}")


def cmd_fundamental(args):
    if not args.symbol and not args.symbols_file:
        print("Provide --symbol SYMBOL or --symbols-file FILE")
        sys.exit(1)
    syms = []
    if args.symbol:
        syms.append(args.symbol)
    if args.symbols_file:
        syms.extend(symbols_mod.load_symbols_from_file(args.symbols_file))
    
    for s in syms:
        try:
            if args.latest:
                # Just print latest ratios
                ratios = get_latest_ratios(s)
                if ratios:
                    print(f"\n=== {s} Latest Ratios ===")
                    for k, v in ratios.items():
                        if v is not None:
                            print(f"  {k}: {v}")
                else:
                    print(f"No fundamental data found for {s}")
            else:
                # Save all to CSV
                paths = save_fundamental_csv(s, out_dir=args.outdir)
                if paths:
                    print(f"Saved fundamental for {s}:")
                    for dtype, path in paths.items():
                        print(f"  {dtype} -> {path}")
                else:
                    print(f"No fundamental data found for {s}")
        except Exception as e:
            print(f"Error fetching fundamental for {s}: {e}")


def main():
    p = argparse.ArgumentParser(description="VN-Index stock data crawler (cafef.vn + TCBS)")
    sub = p.add_subparsers(dest="cmd")

    sp = sub.add_parser("symbols", help="List or fetch stock symbols")
    sp.add_argument("--from-file", help="Path to symbols.txt")
    sp.add_argument("--from-url", help="URL that lists components (cafef)")
    sp.set_defaults(func=cmd_symbols)

    hp = sub.add_parser("historical", help="Fetch historical OHLC data")
    hp.add_argument("--symbol", help="Single symbol to fetch (e.g. VIC, ACV)")
    hp.add_argument("--symbols-file", help="File with symbols, one per line")
    hp.add_argument("--url-template", default=None, help="Optional URL template for HTML fallback (contains {symbol})")
    hp.add_argument("--outdir", default="data/historical", help="Output directory for CSV files")
    hp.set_defaults(func=cmd_historical)

    fp = sub.add_parser("fundamental", help="Fetch fundamental data (P/E, ROE, EPS, etc.)")
    fp.add_argument("--symbol", help="Single symbol to fetch (e.g. VIC, ACV)")
    fp.add_argument("--symbols-file", help="File with symbols, one per line")
    fp.add_argument("--outdir", default="data/fundamental", help="Output directory for CSV files")
    fp.add_argument("--latest", action="store_true", help="Only show latest ratios (don't save to CSV)")
    fp.set_defaults(func=cmd_fundamental)

    args = p.parse_args()
    if not args.cmd:
        p.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
