# Pipeline 
## STAGE 1: CRAWLING DATA
Use crawler to get data of stocks manually or use library `vnstock` for VN-Index stocks.
(other markets: `yfinance` for US stocks, `binance` for crypto, ...)

## STAGE 2: PREPROCESSING AND MANAGING DATA
Preprocess data with `qlib` library 

## STAGE 3: ALPHA FACTORS CREATION AND BACKTEST



# TODO:
- [x] preprocessing data
- [ ] build alpha factors (alpha combination) and backtest with custom formulas 
- [ ] build alpha factors and backtest with machine learning models
- [ ] build portfolio model and backtest

# note for alpha combination:
- low correlation 
- normalize
- different holding periods