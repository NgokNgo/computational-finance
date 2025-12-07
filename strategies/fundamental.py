"""
Fundamental Analysis Trading Strategies
=======================================
This module implements various fundamental-based trading strategies:

1. Value Investing (P/E, P/B, EV/EBITDA)
2. Quality Investing (ROE, ROA, Margins)
3. Growth Investing (Revenue Growth, Earnings Growth)
4. GARP (Growth at Reasonable Price)
5. Piotroski F-Score
6. Dividend Strategy
7. Balance Sheet Strength
8. Free Cash Flow Strategy
9. Composite Multi-Factor Strategy
10. Sector Relative Value

Author: Computational Finance Project
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import from utils modules
from utils.data_loader import (
    load_historical_data,
    load_fundamental_data,
    load_all_fundamentals,
    merge_fundamental_data,
)
from utils.metrics import (
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_cagr,
    calculate_volatility,
)


# =============================================================================
# FUNDAMENTAL METRICS CALCULATION
# =============================================================================

@dataclass
class FundamentalMetrics:
    """Container for fundamental metrics."""
    symbol: str
    year: int
    
    # Valuation
    pe_ratio: float = np.nan
    pb_ratio: float = np.nan
    ev_ebitda: float = np.nan
    dividend_yield: float = np.nan
    
    # Profitability
    roe: float = np.nan
    roa: float = np.nan
    gross_margin: float = np.nan
    operating_margin: float = np.nan
    net_margin: float = np.nan
    
    # Growth
    revenue_growth: float = np.nan
    earnings_growth: float = np.nan
    eps_growth: float = np.nan
    
    # Quality
    current_ratio: float = np.nan
    quick_ratio: float = np.nan
    debt_to_equity: float = np.nan
    interest_coverage: float = np.nan
    
    # Cash Flow
    fcf_yield: float = np.nan
    operating_cf_margin: float = np.nan
    
    # Efficiency
    asset_turnover: float = np.nan
    inventory_days: float = np.nan
    receivable_days: float = np.nan
    
    # Piotroski Score
    f_score: int = 0


def calculate_piotroski_score(fund_df: pd.DataFrame, year: int) -> int:
    """
    Calculate Piotroski F-Score (0-9 points).
    
    Profitability (4 points):
    1. Positive ROA
    2. Positive Operating Cash Flow
    3. ROA improvement vs prior year
    4. Operating CF > Net Income (Accruals)
    
    Leverage/Liquidity (3 points):
    5. Decrease in Debt/Assets
    6. Increase in Current Ratio
    7. No new share issuance
    
    Operating Efficiency (2 points):
    8. Increase in Gross Margin
    9. Increase in Asset Turnover
    """
    score = 0
    
    curr = fund_df[fund_df['year'] == year]
    prev = fund_df[fund_df['year'] == year - 1]
    
    if curr.empty:
        return 0
    
    curr = curr.iloc[-1]
    prev = prev.iloc[-1] if not prev.empty else None
    
    # 1. Positive ROA
    roa = curr.get('roa', 0)
    if not pd.isna(roa) and roa > 0:
        score += 1
    
    # 2. Positive Operating Cash Flow
    ocf = curr.get('fromSale', 0)
    if not pd.isna(ocf) and ocf > 0:
        score += 1
    
    # 3. ROA improvement
    if prev is not None:
        prev_roa = prev.get('roa', 0)
        if not pd.isna(roa) and not pd.isna(prev_roa) and roa > prev_roa:
            score += 1
    
    # 4. Operating CF > Net Income (quality of earnings)
    net_income = curr.get('postTaxProfit', 0)
    if not pd.isna(ocf) and not pd.isna(net_income) and ocf > net_income:
        score += 1
    
    # 5. Decrease in leverage (Debt/Assets)
    debt_asset = curr.get('debtOnAsset', 1)
    if prev is not None:
        prev_debt_asset = prev.get('debtOnAsset', 1)
        if not pd.isna(debt_asset) and not pd.isna(prev_debt_asset) and debt_asset < prev_debt_asset:
            score += 1
    
    # 6. Increase in Current Ratio
    current = curr.get('currentPayment', 0)
    if prev is not None:
        prev_current = prev.get('currentPayment', 0)
        if not pd.isna(current) and not pd.isna(prev_current) and current > prev_current:
            score += 1
    
    # 7. No share dilution
    shares = curr.get('capital', 0)
    if prev is not None:
        prev_shares = prev.get('capital', 0)
        if not pd.isna(shares) and not pd.isna(prev_shares) and shares <= prev_shares:
            score += 1
    
    # 8. Increase in Gross Margin
    gross = curr.get('grossProfitMargin', 0)
    if prev is not None:
        prev_gross = prev.get('grossProfitMargin', 0)
        if not pd.isna(gross) and not pd.isna(prev_gross) and gross > prev_gross:
            score += 1
    
    # 9. Increase in Asset Turnover
    turnover = curr.get('revenueOnAsset', 0)
    if prev is not None:
        prev_turnover = prev.get('revenueOnAsset', 0)
        if not pd.isna(turnover) and not pd.isna(prev_turnover) and turnover > prev_turnover:
            score += 1
    
    return score


# =============================================================================
# FUNDAMENTAL STRATEGIES
# =============================================================================

class FundamentalStrategy:
    """Base class for fundamental strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.scores = None
        
    def score_stocks(self, all_data: Dict[str, Dict[str, pd.DataFrame]], 
                     year: int) -> Dict[str, float]:
        """Score stocks based on fundamental criteria. Override in subclass."""
        raise NotImplementedError
    
    def rank_stocks(self, all_data: Dict[str, Dict[str, pd.DataFrame]], 
                    year: int) -> List[Tuple[str, float]]:
        """Rank stocks from best to worst."""
        scores = self.score_stocks(all_data, year)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class ValueStrategy(FundamentalStrategy):
    """
    Value Investing Strategy.
    Looks for undervalued stocks based on P/E, P/B, EV/EBITDA.
    Lower valuations = higher score.
    """
    
    def __init__(self, pe_weight: float = 0.4, pb_weight: float = 0.3, 
                 ev_ebitda_weight: float = 0.3):
        super().__init__("Value Investing")
        self.pe_weight = pe_weight
        self.pb_weight = pb_weight
        self.ev_ebitda_weight = ev_ebitda_weight
    
    def score_stocks(self, all_data: Dict[str, Dict[str, pd.DataFrame]], 
                     year: int) -> Dict[str, float]:
        scores = {}
        metrics_list = []
        
        for symbol, data in all_data.items():
            merged = merge_fundamental_data(data)
            year_data = merged[merged['year'] == year]
            if year_data.empty:
                continue
            
            row = year_data.iloc[-1]
            pe = row.get('priceToEarning', np.nan)
            pb = row.get('priceToBook', np.nan)
            ev_ebitda = row.get('valueBeforeEbitda', np.nan)
            
            # Filter out invalid/negative values
            if pd.isna(pe) or pe <= 0 or pe > 100:
                pe = np.nan
            if pd.isna(pb) or pb <= 0 or pb > 20:
                pb = np.nan
            if pd.isna(ev_ebitda) or ev_ebitda <= 0 or ev_ebitda > 50:
                ev_ebitda = np.nan
            
            metrics_list.append({
                'symbol': symbol,
                'pe': pe,
                'pb': pb,
                'ev_ebitda': ev_ebitda
            })
        
        if not metrics_list:
            return scores
        
        df = pd.DataFrame(metrics_list)
        
        # Rank (lower is better for value, so invert)
        for col in ['pe', 'pb', 'ev_ebitda']:
            df[f'{col}_score'] = 1 - df[col].rank(pct=True, na_option='bottom')
        
        # Combined score
        df['total_score'] = (
            self.pe_weight * df['pe_score'].fillna(0) +
            self.pb_weight * df['pb_score'].fillna(0) +
            self.ev_ebitda_weight * df['ev_ebitda_score'].fillna(0)
        )
        
        for _, row in df.iterrows():
            scores[row['symbol']] = row['total_score']
        
        return scores


class QualityStrategy(FundamentalStrategy):
    """
    Quality Investing Strategy.
    Looks for high quality companies based on profitability and stability.
    """
    
    def __init__(self, roe_weight: float = 0.3, margin_weight: float = 0.3,
                 stability_weight: float = 0.2, leverage_weight: float = 0.2):
        super().__init__("Quality Investing")
        self.roe_weight = roe_weight
        self.margin_weight = margin_weight
        self.stability_weight = stability_weight
        self.leverage_weight = leverage_weight
    
    def score_stocks(self, all_data: Dict[str, Dict[str, pd.DataFrame]], 
                     year: int) -> Dict[str, float]:
        scores = {}
        metrics_list = []
        
        for symbol, data in all_data.items():
            merged = merge_fundamental_data(data)
            year_data = merged[merged['year'] == year]
            if year_data.empty:
                continue
            
            row = year_data.iloc[-1]
            
            # Get multi-year data for stability
            recent_data = merged[merged['year'].isin(range(year-4, year+1))]
            
            metrics = {
                'symbol': symbol,
                'roe': row.get('roe', np.nan),
                'roa': row.get('roa', np.nan),
                'gross_margin': row.get('grossProfitMargin', np.nan),
                'operating_margin': row.get('operatingProfitMargin', np.nan),
                'debt_equity': row.get('debtOnEquity', np.nan),
                'current_ratio': row.get('currentPayment', np.nan),
                # ROE stability (lower std = more stable)
                'roe_stability': recent_data['roe'].std() if 'roe' in recent_data else np.nan
            }
            metrics_list.append(metrics)
        
        if not metrics_list:
            return scores
        
        df = pd.DataFrame(metrics_list)
        
        # Score each factor
        df['roe_score'] = df['roe'].rank(pct=True, na_option='bottom')
        df['margin_score'] = (
            df['gross_margin'].rank(pct=True, na_option='bottom') * 0.5 +
            df['operating_margin'].rank(pct=True, na_option='bottom') * 0.5
        )
        # Lower debt = higher score
        df['leverage_score'] = 1 - df['debt_equity'].rank(pct=True, na_option='top')
        # Lower volatility = higher stability score
        df['stability_score'] = 1 - df['roe_stability'].rank(pct=True, na_option='top')
        
        df['total_score'] = (
            self.roe_weight * df['roe_score'].fillna(0) +
            self.margin_weight * df['margin_score'].fillna(0) +
            self.stability_weight * df['stability_score'].fillna(0) +
            self.leverage_weight * df['leverage_score'].fillna(0)
        )
        
        for _, row in df.iterrows():
            scores[row['symbol']] = row['total_score']
        
        return scores


class GrowthStrategy(FundamentalStrategy):
    """
    Growth Investing Strategy.
    Looks for companies with strong revenue and earnings growth.
    """
    
    def __init__(self, revenue_weight: float = 0.4, earnings_weight: float = 0.4,
                 eps_weight: float = 0.2):
        super().__init__("Growth Investing")
        self.revenue_weight = revenue_weight
        self.earnings_weight = earnings_weight
        self.eps_weight = eps_weight
    
    def score_stocks(self, all_data: Dict[str, Dict[str, pd.DataFrame]], 
                     year: int) -> Dict[str, float]:
        scores = {}
        metrics_list = []
        
        for symbol, data in all_data.items():
            merged = merge_fundamental_data(data)
            year_data = merged[merged['year'] == year]
            if year_data.empty:
                continue
            
            row = year_data.iloc[-1]
            metrics = {
                'symbol': symbol,
                'revenue_growth': row.get('yearRevenueGrowth', np.nan),
                'earnings_growth': row.get('yearShareHolderIncomeGrowth', np.nan),
                'eps_growth': row.get('epsChange', np.nan)
            }
            metrics_list.append(metrics)
        
        if not metrics_list:
            return scores
        
        df = pd.DataFrame(metrics_list)
        
        # Rank by growth (higher is better)
        df['revenue_score'] = df['revenue_growth'].rank(pct=True, na_option='bottom')
        df['earnings_score'] = df['earnings_growth'].rank(pct=True, na_option='bottom')
        df['eps_score'] = df['eps_growth'].rank(pct=True, na_option='bottom')
        
        df['total_score'] = (
            self.revenue_weight * df['revenue_score'].fillna(0) +
            self.earnings_weight * df['earnings_score'].fillna(0) +
            self.eps_weight * df['eps_score'].fillna(0)
        )
        
        for _, row in df.iterrows():
            scores[row['symbol']] = row['total_score']
        
        return scores


class GARPStrategy(FundamentalStrategy):
    """
    GARP (Growth at Reasonable Price) Strategy.
    Combines growth and value metrics using PEG ratio concept.
    """
    
    def __init__(self, peg_weight: float = 0.4, quality_weight: float = 0.3,
                 momentum_weight: float = 0.3):
        super().__init__("GARP")
        self.peg_weight = peg_weight
        self.quality_weight = quality_weight
        self.momentum_weight = momentum_weight
    
    def score_stocks(self, all_data: Dict[str, Dict[str, pd.DataFrame]], 
                     year: int) -> Dict[str, float]:
        scores = {}
        metrics_list = []
        
        for symbol, data in all_data.items():
            merged = merge_fundamental_data(data)
            year_data = merged[merged['year'] == year]
            if year_data.empty:
                continue
            
            row = year_data.iloc[-1]
            
            pe = row.get('priceToEarning', np.nan)
            earnings_growth = row.get('epsChange', np.nan)
            
            # Calculate PEG ratio
            peg = np.nan
            if not pd.isna(pe) and not pd.isna(earnings_growth) and earnings_growth > 0:
                peg = pe / (earnings_growth * 100)
            
            metrics = {
                'symbol': symbol,
                'peg': peg,
                'roe': row.get('roe', np.nan),
                'earnings_growth': earnings_growth
            }
            metrics_list.append(metrics)
        
        if not metrics_list:
            return scores
        
        df = pd.DataFrame(metrics_list)
        
        # Lower PEG is better (inverted rank)
        df['peg_score'] = 1 - df['peg'].rank(pct=True, na_option='top')
        df['quality_score'] = df['roe'].rank(pct=True, na_option='bottom')
        df['momentum_score'] = df['earnings_growth'].rank(pct=True, na_option='bottom')
        
        df['total_score'] = (
            self.peg_weight * df['peg_score'].fillna(0) +
            self.quality_weight * df['quality_score'].fillna(0) +
            self.momentum_weight * df['momentum_score'].fillna(0)
        )
        
        for _, row in df.iterrows():
            scores[row['symbol']] = row['total_score']
        
        return scores


class PiotroskiStrategy(FundamentalStrategy):
    """
    Piotroski F-Score Strategy.
    Scores companies 0-9 based on profitability, leverage, and efficiency.
    """
    
    def __init__(self, min_score: int = 6):
        super().__init__("Piotroski F-Score")
        self.min_score = min_score
    
    def score_stocks(self, all_data: Dict[str, Dict[str, pd.DataFrame]], 
                     year: int) -> Dict[str, float]:
        scores = {}
        
        for symbol, data in all_data.items():
            merged = merge_fundamental_data(data)
            f_score = calculate_piotroski_score(merged, year)
            scores[symbol] = f_score / 9.0  # Normalize to 0-1
        
        return scores


class DividendStrategy(FundamentalStrategy):
    """
    Dividend Strategy.
    Looks for sustainable high dividend yields.
    """
    
    def __init__(self, yield_weight: float = 0.5, sustainability_weight: float = 0.3,
                 growth_weight: float = 0.2):
        super().__init__("Dividend Strategy")
        self.yield_weight = yield_weight
        self.sustainability_weight = sustainability_weight
        self.growth_weight = growth_weight
    
    def score_stocks(self, all_data: Dict[str, Dict[str, pd.DataFrame]], 
                     year: int) -> Dict[str, float]:
        scores = {}
        metrics_list = []
        
        for symbol, data in all_data.items():
            merged = merge_fundamental_data(data)
            year_data = merged[merged['year'] == year]
            if year_data.empty:
                continue
            
            row = year_data.iloc[-1]
            div_yield = row.get('dividend', np.nan)
            fcf = row.get('freeCashFlow', np.nan)
            
            # Dividend growth
            prev_data = merged[merged['year'] == year - 1]
            div_growth = np.nan
            if not prev_data.empty:
                prev_div = prev_data.iloc[-1].get('dividend', np.nan)
                if not pd.isna(prev_div) and not pd.isna(div_yield) and prev_div > 0:
                    div_growth = (div_yield - prev_div) / prev_div
            
            metrics_list.append({
                'symbol': symbol,
                'div_yield': div_yield,
                'fcf': fcf,
                'div_growth': div_growth
            })
        
        if not metrics_list:
            return scores
        
        df = pd.DataFrame(metrics_list)
        
        df['yield_score'] = df['div_yield'].rank(pct=True, na_option='bottom')
        df['fcf_score'] = df['fcf'].rank(pct=True, na_option='bottom')
        df['growth_score'] = df['div_growth'].rank(pct=True, na_option='bottom')
        
        df['total_score'] = (
            self.yield_weight * df['yield_score'].fillna(0) +
            self.sustainability_weight * df['fcf_score'].fillna(0) +
            self.growth_weight * df['growth_score'].fillna(0)
        )
        
        for _, row in df.iterrows():
            scores[row['symbol']] = row['total_score']
        
        return scores


class BalanceSheetStrategy(FundamentalStrategy):
    """
    Balance Sheet Strength Strategy.
    Focuses on financial health and stability.
    """
    
    def __init__(self):
        super().__init__("Balance Sheet Strength")
    
    def score_stocks(self, all_data: Dict[str, Dict[str, pd.DataFrame]], 
                     year: int) -> Dict[str, float]:
        scores = {}
        metrics_list = []
        
        for symbol, data in all_data.items():
            merged = merge_fundamental_data(data)
            year_data = merged[merged['year'] == year]
            if year_data.empty:
                continue
            
            row = year_data.iloc[-1]
            
            metrics = {
                'symbol': symbol,
                'current_ratio': row.get('currentPayment', np.nan),
                'quick_ratio': row.get('quickPayment', np.nan),
                'debt_equity': row.get('debtOnEquity', np.nan),
                'debt_asset': row.get('debtOnAsset', np.nan),
                'interest_coverage': row.get('ebitOnInterest', np.nan),
                'equity_ratio': row.get('equityOnTotalAsset', np.nan)
            }
            metrics_list.append(metrics)
        
        if not metrics_list:
            return scores
        
        df = pd.DataFrame(metrics_list)
        
        # Higher is better
        df['current_score'] = df['current_ratio'].rank(pct=True, na_option='bottom')
        df['quick_score'] = df['quick_ratio'].rank(pct=True, na_option='bottom')
        df['coverage_score'] = df['interest_coverage'].rank(pct=True, na_option='bottom')
        df['equity_score'] = df['equity_ratio'].rank(pct=True, na_option='bottom')
        
        # Lower is better (inverted)
        df['debt_eq_score'] = 1 - df['debt_equity'].rank(pct=True, na_option='top')
        df['debt_asset_score'] = 1 - df['debt_asset'].rank(pct=True, na_option='top')
        
        df['total_score'] = (
            df['current_score'].fillna(0) * 0.15 +
            df['quick_score'].fillna(0) * 0.15 +
            df['debt_eq_score'].fillna(0) * 0.2 +
            df['debt_asset_score'].fillna(0) * 0.2 +
            df['coverage_score'].fillna(0) * 0.15 +
            df['equity_score'].fillna(0) * 0.15
        )
        
        for _, row in df.iterrows():
            scores[row['symbol']] = row['total_score']
        
        return scores


class FCFStrategy(FundamentalStrategy):
    """
    Free Cash Flow Strategy.
    Focuses on cash generation ability.
    """
    
    def __init__(self, fcf_yield_weight: float = 0.4, fcf_margin_weight: float = 0.3,
                 fcf_growth_weight: float = 0.3):
        super().__init__("Free Cash Flow")
        self.fcf_yield_weight = fcf_yield_weight
        self.fcf_margin_weight = fcf_margin_weight
        self.fcf_growth_weight = fcf_growth_weight
    
    def score_stocks(self, all_data: Dict[str, Dict[str, pd.DataFrame]], 
                     year: int) -> Dict[str, float]:
        scores = {}
        metrics_list = []
        
        for symbol, data in all_data.items():
            merged = merge_fundamental_data(data)
            year_data = merged[merged['year'] == year]
            if year_data.empty:
                continue
            
            row = year_data.iloc[-1]
            
            fcf = row.get('freeCashFlow', np.nan)
            equity = row.get('equity', np.nan)
            revenue = row.get('revenue', np.nan)
            
            fcf_yield = fcf / equity if equity and equity > 0 else np.nan
            fcf_margin = fcf / revenue if revenue and revenue > 0 else np.nan
            
            # FCF growth
            prev_data = merged[merged['year'] == year - 1]
            fcf_growth = np.nan
            if not prev_data.empty:
                prev_fcf = prev_data.iloc[-1].get('freeCashFlow', np.nan)
                if not pd.isna(prev_fcf) and prev_fcf > 0:
                    fcf_growth = (fcf - prev_fcf) / prev_fcf
            
            metrics_list.append({
                'symbol': symbol,
                'fcf_yield': fcf_yield,
                'fcf_margin': fcf_margin,
                'fcf_growth': fcf_growth
            })
        
        if not metrics_list:
            return scores
        
        df = pd.DataFrame(metrics_list)
        
        df['yield_score'] = df['fcf_yield'].rank(pct=True, na_option='bottom')
        df['margin_score'] = df['fcf_margin'].rank(pct=True, na_option='bottom')
        df['growth_score'] = df['fcf_growth'].rank(pct=True, na_option='bottom')
        
        df['total_score'] = (
            self.fcf_yield_weight * df['yield_score'].fillna(0) +
            self.fcf_margin_weight * df['margin_score'].fillna(0) +
            self.fcf_growth_weight * df['growth_score'].fillna(0)
        )
        
        for _, row in df.iterrows():
            scores[row['symbol']] = row['total_score']
        
        return scores


class CompositeStrategy(FundamentalStrategy):
    """
    Composite Multi-Factor Strategy.
    Combines Value, Quality, Growth, and Momentum factors.
    """
    
    def __init__(self, value_weight: float = 0.25, quality_weight: float = 0.25,
                 growth_weight: float = 0.25, momentum_weight: float = 0.25):
        super().__init__("Composite Multi-Factor")
        self.value_weight = value_weight
        self.quality_weight = quality_weight
        self.growth_weight = growth_weight
        self.momentum_weight = momentum_weight
        
        self.value_strategy = ValueStrategy()
        self.quality_strategy = QualityStrategy()
        self.growth_strategy = GrowthStrategy()
    
    def score_stocks(self, all_data: Dict[str, Dict[str, pd.DataFrame]], 
                     year: int) -> Dict[str, float]:
        # Get individual strategy scores
        value_scores = self.value_strategy.score_stocks(all_data, year)
        quality_scores = self.quality_strategy.score_stocks(all_data, year)
        growth_scores = self.growth_strategy.score_stocks(all_data, year)
        
        # Combine scores
        all_symbols = set(value_scores.keys()) | set(quality_scores.keys()) | set(growth_scores.keys())
        
        scores = {}
        for symbol in all_symbols:
            v = value_scores.get(symbol, 0)
            q = quality_scores.get(symbol, 0)
            g = growth_scores.get(symbol, 0)
            
            scores[symbol] = (
                self.value_weight * v +
                self.quality_weight * q +
                self.growth_weight * g
            )
        
        return scores


class SectorRelativeStrategy(FundamentalStrategy):
    """
    Sector Relative Value Strategy.
    Compares stocks within sectors.
    """
    
    def __init__(self):
        super().__init__("Sector Relative Value")
    
    def score_stocks(self, all_data: Dict[str, Dict[str, pd.DataFrame]], 
                     year: int) -> Dict[str, float]:
        # Simplified - uses overall ranking without sector classification
        value = ValueStrategy()
        return value.score_stocks(all_data, year)


# =============================================================================
# PORTFOLIO CONSTRUCTION
# =============================================================================

class FundamentalPortfolio:
    """
    Construct portfolios based on fundamental strategies.
    """
    
    def __init__(self, strategy: FundamentalStrategy, top_n: int = 3, 
                 rebalance_freq: str = 'annual'):
        self.strategy = strategy
        self.top_n = top_n
        self.rebalance_freq = rebalance_freq
    
    def construct_portfolio(self, all_data: Dict[str, Dict[str, pd.DataFrame]], 
                            year: int) -> Dict[str, float]:
        """Construct equal-weight portfolio of top stocks."""
        rankings = self.strategy.rank_stocks(all_data, year)
        
        # Take top N stocks
        top_stocks = rankings[:self.top_n]
        
        if not top_stocks:
            return {}
        
        # Equal weight
        weight = 1.0 / len(top_stocks)
        return {symbol: weight for symbol, _ in top_stocks}


# =============================================================================
# BACKTESTING
# =============================================================================

def backtest_fundamental_strategy(strategy: FundamentalStrategy,
                                   all_data: Dict[str, Dict[str, pd.DataFrame]],
                                   historical_data: Dict[str, pd.DataFrame],
                                   start_year: int = 2020,
                                   end_year: int = 2024,
                                   top_n: int = 3) -> Dict:
    """
    Backtest a fundamental strategy.
    
    Args:
        strategy: FundamentalStrategy instance
        all_data: Fundamental data for all symbols
        historical_data: Historical price data for all symbols
        start_year: Start year for backtest
        end_year: End year for backtest
        top_n: Number of top stocks to hold
        
    Returns:
        Dictionary with backtest results
    """
    portfolio = FundamentalPortfolio(strategy, top_n=top_n)
    
    results = {
        'strategy': strategy.name,
        'years': [],
        'portfolios': [],
        'returns': []
    }
    
    for year in range(start_year, end_year + 1):
        weights = portfolio.construct_portfolio(all_data, year)
        results['years'].append(year)
        results['portfolios'].append(weights)
    
    return results


__all__ = [
    'FundamentalMetrics',
    'FundamentalStrategy',
    'ValueStrategy',
    'QualityStrategy',
    'GrowthStrategy',
    'GARPStrategy',
    'PiotroskiStrategy',
    'DividendStrategy',
    'BalanceSheetStrategy',
    'FCFStrategy',
    'CompositeStrategy',
    'SectorRelativeStrategy',
    'FundamentalPortfolio',
    'calculate_piotroski_score',
    'backtest_fundamental_strategy',
]
