import ccxt
import pandas as pd
import os
import time

exchange = ccxt.binance()

symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 
           'DOGE/USDT', 'SHIB/USDT', 
           'TIA/USDT', 'RNDR/USDT']

timeframe = '1d'
limit = 1000
since = exchange.parse8601('2015-01-01T00:00:00Z')
save_dir = 'data'

os.makedirs(save_dir, exist_ok=True)

def fetch_ohlc(symbol):
    print(f"\n📈 Fetching {symbol} {timeframe} data...")
    all_data = []
    since_local = since

    while True:
        ohlcvs = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_local, limit=limit)
        if not ohlcvs:
            break
        all_data += ohlcvs
        since_local = ohlcvs[-1][0] + 1
        print(f"  + {len(ohlcvs)} candles, total: {len(all_data)}")

        time.sleep(exchange.rateLimit / 1000)

        # optional stop after ~10 years
        if len(all_data) >= 3650:
            break

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    return df

# Lặp qua từng mã
for sym in symbols:
    try:
        df = fetch_ohlc(sym)
        filename = sym.replace('/', '') + f'_{timeframe}.csv'
        path = os.path.join(save_dir, filename)
        df.to_csv(path, index=False)
        print(f"✅ Saved {len(df)} rows to {path}")
    except Exception as e:
        print(f"❌ Error with {sym}: {e}")
        time.sleep(5)
