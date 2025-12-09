# Pipeline 
1. CRAWLING DATA: Use crawler to get data of stocks manually or use library `vnstock` for VN-Index stocks. (other markets: `yfinance` for US stocks, `binance` for crypto, ...)
2. PREPROCESSING AND MANAGING DATA with `qlib` library
3. ALPHA FACTORS CREATION AND BACKTEST
4. PORTFOLIO MODELING AND BACKTEST


# TODO:
- [ ] run model-based strategies and optimize hyperparameters with walk-forward validation
- [ ] implement optimizer for portfolio construction 
- [ ] combined alpha factors
- [ ] implement more advanced strategies (XGBoost, LSTM,...)
- [ ] Agent for strategy selection and portfolio rebalancing

# note for alpha combination:
- low correlation 
- normalize
- different holding periods