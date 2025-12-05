"""
Alpha Combiner - Framework for Composite Alpha Generation
==========================================================
Base code for combining multiple alpha signals into a composite alpha.

Features:
- Multiple combination methods (equal weight, IC-weighted, optimization)
- Alpha preprocessing (winsorization, neutralization, standardization)
- Performance tracking and IC analysis
- Portfolio construction from combined alphas

Author: Computational Finance Project
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# ENUMS AND CONFIGURATIONS
# =============================================================================

class CombineMethod(Enum):
    """Methods for combining alpha signals."""
    EQUAL_WEIGHT = "equal_weight"
    IC_WEIGHTED = "ic_weighted"
    IR_WEIGHTED = "ir_weighted"
    RANK_WEIGHTED = "rank_weighted"
    OPTIMIZATION = "optimization"
    MACHINE_LEARNING = "ml"


class NeutralizationMethod(Enum):
    """Methods for alpha neutralization."""
    NONE = "none"
    MARKET = "market"
    SECTOR = "sector"
    MARKET_AND_SECTOR = "market_and_sector"


@dataclass
class AlphaConfig:
    """Configuration for individual alpha."""
    name: str
    weight: float = 1.0
    decay: int = 0  # Days to decay the alpha (0 = no decay)
    delay: int = 1  # Trading delay in days
    neutralize: NeutralizationMethod = NeutralizationMethod.NONE
    winsorize: Tuple[float, float] = (0.01, 0.99)  # Percentile bounds
    enabled: bool = True


@dataclass
class AlphaResult:
    """Container for alpha computation results."""
    name: str
    values: pd.DataFrame  # DataFrame with dates as index, symbols as columns
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ir: float = 0.0  # Information Ratio = IC_mean / IC_std
    turnover: float = 0.0
    coverage: float = 0.0


# =============================================================================
# BASE ALPHA CLASS
# =============================================================================

class BaseAlpha(ABC):
    """
    Abstract base class for all alpha factors.
    
    Subclass this to create custom alpha factors.
    """
    
    def __init__(self, name: str, config: Optional[AlphaConfig] = None):
        self.name = name
        self.config = config or AlphaConfig(name=name)
        self._cache: Dict[str, pd.DataFrame] = {}
    
    @abstractmethod
    def compute(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compute alpha values.
        
        Args:
            data: Dictionary containing price/fundamental data
                  Expected keys: 'prices' (DataFrame with OHLCV)
                  Optional: 'fundamentals', 'sector', etc.
        
        Returns:
            DataFrame with dates as index, symbols as columns
        """
        pass
    
    def preprocess(self, alpha: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing steps to alpha values."""
        # Winsorize
        if self.config.winsorize:
            alpha = self._winsorize(alpha, *self.config.winsorize)
        
        # Apply delay
        if self.config.delay > 0:
            alpha = alpha.shift(self.config.delay)
        
        # Apply decay
        if self.config.decay > 0:
            alpha = self._apply_decay(alpha, self.config.decay)
        
        # Standardize cross-sectionally
        alpha = self._standardize(alpha)
        
        return alpha
    
    def _winsorize(self, df: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
        """Winsorize values to percentile bounds."""
        def winsorize_row(row):
            if row.isna().all():
                return row
            lower_val = row.quantile(lower)
            upper_val = row.quantile(upper)
            return row.clip(lower=lower_val, upper=upper_val)
        
        return df.apply(winsorize_row, axis=1)
    
    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional standardization (z-score)."""
        mean = df.mean(axis=1)
        std = df.std(axis=1)
        return df.sub(mean, axis=0).div(std, axis=0)
    
    def _apply_decay(self, df: pd.DataFrame, halflife: int) -> pd.DataFrame:
        """Apply exponential decay to alpha values."""
        return df.ewm(halflife=halflife, min_periods=1).mean()
    
    def get_result(self, data: Dict[str, pd.DataFrame], forward_returns: pd.DataFrame) -> AlphaResult:
        """Compute alpha and calculate performance metrics."""
        # Compute raw alpha
        alpha_raw = self.compute(data)
        
        # Preprocess
        alpha = self.preprocess(alpha_raw)
        
        # Calculate IC
        ic_series = self._calculate_ic_series(alpha, forward_returns)
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        ir = ic_mean / ic_std if ic_std > 0 else 0
        
        # Calculate turnover
        turnover = self._calculate_turnover(alpha)
        
        # Calculate coverage
        coverage = alpha.notna().mean().mean()
        
        return AlphaResult(
            name=self.name,
            values=alpha,
            ic_mean=ic_mean,
            ic_std=ic_std,
            ir=ir,
            turnover=turnover,
            coverage=coverage
        )
    
    def _calculate_ic_series(self, alpha: pd.DataFrame, forward_returns: pd.DataFrame) -> pd.Series:
        """Calculate Information Coefficient (rank correlation) time series."""
        common_dates = alpha.index.intersection(forward_returns.index)
        common_cols = alpha.columns.intersection(forward_returns.columns)
        
        ic_list = []
        for date in common_dates:
            a = alpha.loc[date, common_cols]
            r = forward_returns.loc[date, common_cols]
            
            mask = a.notna() & r.notna()
            if mask.sum() > 5:
                ic = a[mask].corr(r[mask], method='spearman')
                ic_list.append(ic)
            else:
                ic_list.append(np.nan)
        
        return pd.Series(ic_list, index=common_dates)
    
    def _calculate_turnover(self, alpha: pd.DataFrame) -> float:
        """Calculate average daily turnover."""
        # Rank-based turnover
        ranks = alpha.rank(axis=1, pct=True)
        turnover = ranks.diff().abs().mean(axis=1).mean()
        return turnover


# =============================================================================
# BUILT-IN ALPHA FACTORS
# =============================================================================

class MomentumAlpha(BaseAlpha):
    """Price momentum alpha factor."""
    
    def __init__(self, lookback: int = 20, name: str = None):
        super().__init__(name or f"momentum_{lookback}d")
        self.lookback = lookback
    
    def compute(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        prices = data['prices']
        if isinstance(prices, pd.DataFrame) and 'close' in prices.columns:
            # Single stock format
            close = prices.pivot_table(index='date', columns='symbol', values='close')
        else:
            close = prices
        
        returns = close.pct_change(self.lookback)
        return returns


class ReversalAlpha(BaseAlpha):
    """Short-term reversal alpha factor."""
    
    def __init__(self, lookback: int = 5, name: str = None):
        super().__init__(name or f"reversal_{lookback}d")
        self.lookback = lookback
    
    def compute(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        prices = data['prices']
        if isinstance(prices, pd.DataFrame) and 'close' in prices.columns:
            close = prices.pivot_table(index='date', columns='symbol', values='close')
        else:
            close = prices
        
        returns = close.pct_change(self.lookback)
        return -returns  # Negative for reversal


class VolatilityAlpha(BaseAlpha):
    """Low volatility alpha factor."""
    
    def __init__(self, lookback: int = 20, name: str = None):
        super().__init__(name or f"low_vol_{lookback}d")
        self.lookback = lookback
    
    def compute(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        prices = data['prices']
        if isinstance(prices, pd.DataFrame) and 'close' in prices.columns:
            close = prices.pivot_table(index='date', columns='symbol', values='close')
        else:
            close = prices
        
        daily_returns = close.pct_change()
        volatility = daily_returns.rolling(self.lookback).std()
        return -volatility  # Negative because low vol is preferred


class VolumeAlpha(BaseAlpha):
    """Volume momentum alpha factor."""
    
    def __init__(self, lookback: int = 20, name: str = None):
        super().__init__(name or f"volume_{lookback}d")
        self.lookback = lookback
    
    def compute(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        prices = data['prices']
        if isinstance(prices, pd.DataFrame) and 'volume' in prices.columns:
            volume = prices.pivot_table(index='date', columns='symbol', values='volume')
        else:
            volume = data.get('volume', prices)
        
        vol_ma = volume.rolling(self.lookback).mean()
        vol_ratio = volume / vol_ma
        return vol_ratio


class ValueAlpha(BaseAlpha):
    """Value alpha from fundamental data (1/PE)."""
    
    def __init__(self, name: str = "value_pe"):
        super().__init__(name)
    
    def compute(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        fundamentals = data.get('fundamentals')
        if fundamentals is None:
            raise ValueError("Fundamentals data required for ValueAlpha")
        
        if 'pe' in fundamentals.columns:
            pe = fundamentals.pivot_table(index='date', columns='symbol', values='pe')
        else:
            pe = fundamentals
        
        # Earnings yield = 1/PE (higher is better for value)
        earnings_yield = 1 / pe.replace(0, np.nan)
        return earnings_yield


class QualityAlpha(BaseAlpha):
    """Quality alpha from ROE."""
    
    def __init__(self, name: str = "quality_roe"):
        super().__init__(name)
    
    def compute(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        fundamentals = data.get('fundamentals')
        if fundamentals is None:
            raise ValueError("Fundamentals data required for QualityAlpha")
        
        if 'roe' in fundamentals.columns:
            roe = fundamentals.pivot_table(index='date', columns='symbol', values='roe')
        else:
            roe = fundamentals
        
        return roe


# =============================================================================
# ALPHA COMBINER
# =============================================================================

class AlphaCombiner:
    """
    Combines multiple alpha factors into a composite alpha signal.
    
    Usage:
        combiner = AlphaCombiner()
        combiner.add_alpha(MomentumAlpha(20))
        combiner.add_alpha(ReversalAlpha(5))
        combiner.add_alpha(VolatilityAlpha(20))
        
        composite = combiner.combine(data, method=CombineMethod.IC_WEIGHTED)
    """
    
    def __init__(self):
        self.alphas: List[BaseAlpha] = []
        self.alpha_results: Dict[str, AlphaResult] = {}
        self.weights: Dict[str, float] = {}
    
    def add_alpha(self, alpha: BaseAlpha, weight: float = 1.0) -> 'AlphaCombiner':
        """Add an alpha factor to the combiner."""
        self.alphas.append(alpha)
        self.weights[alpha.name] = weight
        return self
    
    def remove_alpha(self, name: str) -> 'AlphaCombiner':
        """Remove an alpha factor by name."""
        self.alphas = [a for a in self.alphas if a.name != name]
        self.weights.pop(name, None)
        return self
    
    def compute_all(self, data: Dict[str, pd.DataFrame], forward_returns: pd.DataFrame) -> Dict[str, AlphaResult]:
        """Compute all alpha factors and their metrics."""
        self.alpha_results = {}
        
        for alpha in self.alphas:
            if alpha.config.enabled:
                result = alpha.get_result(data, forward_returns)
                self.alpha_results[alpha.name] = result
        
        return self.alpha_results
    
    def combine(
        self, 
        data: Dict[str, pd.DataFrame],
        forward_returns: Optional[pd.DataFrame] = None,
        method: CombineMethod = CombineMethod.EQUAL_WEIGHT,
        lookback_ic: int = 60
    ) -> pd.DataFrame:
        """
        Combine alpha factors into composite alpha.
        
        Args:
            data: Market data dictionary
            forward_returns: Forward returns for IC calculation (required for IC/IR weighting)
            method: Combination method
            lookback_ic: Lookback period for IC calculation
            
        Returns:
            DataFrame with composite alpha values
        """
        # Compute all alphas if not already done
        if not self.alpha_results:
            if forward_returns is None:
                # Use simple returns as proxy
                prices = data['prices']
                if isinstance(prices, pd.DataFrame) and 'close' in prices.columns:
                    close = prices.pivot_table(index='date', columns='symbol', values='close')
                else:
                    close = prices
                forward_returns = close.pct_change().shift(-1)
            
            self.compute_all(data, forward_returns)
        
        # Get alpha values
        alpha_dfs = {name: result.values for name, result in self.alpha_results.items()}
        
        if not alpha_dfs:
            raise ValueError("No alpha factors computed")
        
        # Combine based on method
        if method == CombineMethod.EQUAL_WEIGHT:
            return self._combine_equal_weight(alpha_dfs)
        
        elif method == CombineMethod.IC_WEIGHTED:
            return self._combine_ic_weighted(alpha_dfs, forward_returns, lookback_ic)
        
        elif method == CombineMethod.IR_WEIGHTED:
            return self._combine_ir_weighted(alpha_dfs)
        
        elif method == CombineMethod.RANK_WEIGHTED:
            return self._combine_rank_weighted(alpha_dfs)
        
        elif method == CombineMethod.OPTIMIZATION:
            return self._combine_optimization(alpha_dfs, forward_returns)
        
        else:
            raise ValueError(f"Unknown combination method: {method}")
    
    def _combine_equal_weight(self, alpha_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Equal-weighted combination."""
        # Align all alphas
        aligned = self._align_alphas(alpha_dfs)
        
        # Apply user-specified weights
        weighted_sum = None
        total_weight = 0
        
        for name, df in aligned.items():
            w = self.weights.get(name, 1.0)
            if weighted_sum is None:
                weighted_sum = df * w
            else:
                weighted_sum = weighted_sum.add(df * w, fill_value=0)
            total_weight += w
        
        return weighted_sum / total_weight
    
    def _combine_ic_weighted(
        self, 
        alpha_dfs: Dict[str, pd.DataFrame], 
        forward_returns: pd.DataFrame,
        lookback: int = 60
    ) -> pd.DataFrame:
        """IC-weighted combination using rolling IC."""
        aligned = self._align_alphas(alpha_dfs)
        
        # Calculate rolling IC for each alpha
        ic_weights = {}
        for name, alpha_df in aligned.items():
            ic_series = self._rolling_ic(alpha_df, forward_returns, lookback)
            ic_weights[name] = ic_series.clip(lower=0)  # Only positive IC
        
        # Combine with IC weights
        composite = None
        
        for date in aligned[list(aligned.keys())[0]].index:
            row_sum = None
            weight_sum = 0
            
            for name, alpha_df in aligned.items():
                if date in alpha_df.index and date in ic_weights[name].index:
                    ic = ic_weights[name].loc[date]
                    if not np.isnan(ic) and ic > 0:
                        alpha_row = alpha_df.loc[date]
                        if row_sum is None:
                            row_sum = alpha_row * ic
                        else:
                            row_sum = row_sum.add(alpha_row * ic, fill_value=0)
                        weight_sum += ic
            
            if row_sum is not None and weight_sum > 0:
                row_sum = row_sum / weight_sum
                if composite is None:
                    composite = pd.DataFrame([row_sum], index=[date])
                else:
                    composite = pd.concat([composite, pd.DataFrame([row_sum], index=[date])])
        
        return composite if composite is not None else pd.DataFrame()
    
    def _combine_ir_weighted(self, alpha_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """IR-weighted combination using precomputed IR."""
        aligned = self._align_alphas(alpha_dfs)
        
        weighted_sum = None
        total_weight = 0
        
        for name, df in aligned.items():
            ir = self.alpha_results[name].ir
            w = max(ir, 0)  # Only positive IR
            
            if w > 0:
                if weighted_sum is None:
                    weighted_sum = df * w
                else:
                    weighted_sum = weighted_sum.add(df * w, fill_value=0)
                total_weight += w
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return self._combine_equal_weight(alpha_dfs)
    
    def _combine_rank_weighted(self, alpha_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine using rank averages."""
        aligned = self._align_alphas(alpha_dfs)
        
        # Convert to ranks
        rank_dfs = {}
        for name, df in aligned.items():
            rank_dfs[name] = df.rank(axis=1, pct=True)
        
        # Average ranks
        rank_sum = None
        for name, df in rank_dfs.items():
            w = self.weights.get(name, 1.0)
            if rank_sum is None:
                rank_sum = df * w
            else:
                rank_sum = rank_sum.add(df * w, fill_value=0)
        
        return rank_sum / sum(self.weights.values())
    
    def _combine_optimization(
        self, 
        alpha_dfs: Dict[str, pd.DataFrame], 
        forward_returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Optimize weights to maximize IR.
        Uses simple grid search for demo - can be replaced with scipy.optimize.
        """
        aligned = self._align_alphas(alpha_dfs)
        names = list(aligned.keys())
        n = len(names)
        
        if n == 1:
            return list(aligned.values())[0]
        
        # Grid search for optimal weights
        best_ir = -np.inf
        best_weights = {name: 1/n for name in names}
        
        # Simple grid: try different weight combinations
        for i in range(11):
            for j in range(11 - i):
                k = 10 - i - j
                if n == 2:
                    weights = {names[0]: i/10, names[1]: (10-i)/10}
                elif n >= 3:
                    weights = {names[0]: i/10, names[1]: j/10}
                    remaining = k / 10
                    for idx in range(2, n):
                        weights[names[idx]] = remaining / (n - 2)
                else:
                    continue
                
                # Calculate composite
                composite = None
                for name, df in aligned.items():
                    w = weights[name]
                    if composite is None:
                        composite = df * w
                    else:
                        composite = composite.add(df * w, fill_value=0)
                
                # Calculate IR
                ic_series = self._calculate_ic_series_static(composite, forward_returns)
                ir = ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0
                
                if ir > best_ir:
                    best_ir = ir
                    best_weights = weights.copy()
        
        # Apply best weights
        self.weights = best_weights
        return self._combine_equal_weight(aligned)
    
    def _align_alphas(self, alpha_dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Align alpha DataFrames to common dates and symbols."""
        if not alpha_dfs:
            return {}
        
        # Find common dates and symbols
        common_dates = None
        common_symbols = None
        
        for df in alpha_dfs.values():
            if common_dates is None:
                common_dates = set(df.index)
                common_symbols = set(df.columns)
            else:
                common_dates &= set(df.index)
                common_symbols &= set(df.columns)
        
        common_dates = sorted(common_dates)
        common_symbols = sorted(common_symbols)
        
        # Align
        aligned = {}
        for name, df in alpha_dfs.items():
            aligned[name] = df.loc[common_dates, common_symbols]
        
        return aligned
    
    def _rolling_ic(self, alpha: pd.DataFrame, returns: pd.DataFrame, window: int) -> pd.Series:
        """Calculate rolling IC."""
        common_dates = alpha.index.intersection(returns.index)
        common_cols = alpha.columns.intersection(returns.columns)
        
        ic_list = []
        for i, date in enumerate(common_dates):
            if i < window:
                ic_list.append(np.nan)
                continue
            
            # Use past window days for IC calculation
            start_idx = max(0, i - window)
            window_dates = common_dates[start_idx:i]
            
            ics = []
            for d in window_dates:
                a = alpha.loc[d, common_cols]
                r = returns.loc[d, common_cols]
                mask = a.notna() & r.notna()
                if mask.sum() > 5:
                    ic = a[mask].corr(r[mask], method='spearman')
                    ics.append(ic)
            
            ic_list.append(np.mean(ics) if ics else np.nan)
        
        return pd.Series(ic_list, index=common_dates)
    
    @staticmethod
    def _calculate_ic_series_static(alpha: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
        """Static method to calculate IC series."""
        common_dates = alpha.index.intersection(returns.index)
        common_cols = alpha.columns.intersection(returns.columns)
        
        ic_list = []
        for date in common_dates:
            a = alpha.loc[date, common_cols]
            r = returns.loc[date, common_cols]
            mask = a.notna() & r.notna()
            if mask.sum() > 5:
                ic = a[mask].corr(r[mask], method='spearman')
                ic_list.append(ic)
            else:
                ic_list.append(np.nan)
        
        return pd.Series(ic_list, index=common_dates)
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary of all alpha factors."""
        if not self.alpha_results:
            return pd.DataFrame()
        
        rows = []
        for name, result in self.alpha_results.items():
            rows.append({
                'Alpha': name,
                'IC Mean': result.ic_mean,
                'IC Std': result.ic_std,
                'IR': result.ir,
                'Turnover': result.turnover,
                'Coverage': result.coverage,
                'Weight': self.weights.get(name, 1.0)
            })
        
        return pd.DataFrame(rows).set_index('Alpha')
    
    def plot_ic_comparison(self, figsize: Tuple[int, int] = (12, 6)):
        """Plot IC comparison across alphas."""
        import matplotlib.pyplot as plt
        
        summary = self.get_summary()
        if summary.empty:
            print("No alpha results to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # IC Mean comparison
        ax1 = axes[0]
        summary['IC Mean'].plot(kind='bar', ax=ax1, color='steelblue')
        ax1.set_title('IC Mean by Alpha')
        ax1.set_ylabel('IC Mean')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.tick_params(axis='x', rotation=45)
        
        # IR comparison
        ax2 = axes[1]
        summary['IR'].plot(kind='bar', ax=ax2, color='darkgreen')
        ax2.set_title('Information Ratio by Alpha')
        ax2.set_ylabel('IR')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()


# =============================================================================
# PORTFOLIO CONSTRUCTOR
# =============================================================================

class AlphaPortfolio:
    """
    Construct portfolio from composite alpha.
    
    Supports:
    - Long-only or long-short portfolios
    - Top/bottom N stocks or quantile-based
    - Position sizing (equal weight, alpha-weighted)
    """
    
    def __init__(
        self,
        n_long: int = 10,
        n_short: int = 0,
        weight_method: str = 'equal',  # 'equal' or 'alpha'
        max_weight: float = 0.2,
        rebalance_freq: str = 'daily'  # 'daily', 'weekly', 'monthly'
    ):
        self.n_long = n_long
        self.n_short = n_short
        self.weight_method = weight_method
        self.max_weight = max_weight
        self.rebalance_freq = rebalance_freq
    
    def construct(self, alpha: pd.DataFrame) -> pd.DataFrame:
        """
        Construct portfolio weights from alpha signals.
        
        Args:
            alpha: DataFrame with alpha values (dates x symbols)
            
        Returns:
            DataFrame with portfolio weights (dates x symbols)
        """
        weights = pd.DataFrame(0.0, index=alpha.index, columns=alpha.columns)
        
        # Determine rebalance dates
        if self.rebalance_freq == 'weekly':
            # Rebalance on Mondays
            rebal_mask = alpha.index.dayofweek == 0
        elif self.rebalance_freq == 'monthly':
            # Rebalance on first day of month
            rebal_mask = alpha.index.is_month_start
        else:
            # Daily rebalancing
            rebal_mask = pd.Series(True, index=alpha.index)
        
        current_weights = pd.Series(0.0, index=alpha.columns)
        
        for date in alpha.index:
            if rebal_mask.loc[date] if isinstance(rebal_mask, pd.Series) else rebal_mask:
                # Rebalance
                alpha_row = alpha.loc[date].dropna()
                
                if len(alpha_row) >= self.n_long:
                    # Rank stocks
                    ranked = alpha_row.rank(ascending=False)
                    
                    # Long positions
                    long_stocks = ranked.nsmallest(self.n_long).index.tolist()
                    
                    # Short positions
                    short_stocks = []
                    if self.n_short > 0:
                        short_stocks = ranked.nlargest(self.n_short).index.tolist()
                    
                    # Calculate weights
                    if self.weight_method == 'equal':
                        long_weight = 1.0 / self.n_long if self.n_long > 0 else 0
                        short_weight = -1.0 / self.n_short if self.n_short > 0 else 0
                    else:
                        # Alpha-weighted
                        long_alphas = alpha_row[long_stocks]
                        long_weight_total = long_alphas / long_alphas.sum()
                        
                        if short_stocks:
                            short_alphas = -alpha_row[short_stocks]
                            short_weight_total = short_alphas / short_alphas.sum()
                    
                    # Build weight series
                    current_weights = pd.Series(0.0, index=alpha.columns)
                    
                    for stock in long_stocks:
                        if self.weight_method == 'equal':
                            current_weights[stock] = min(long_weight, self.max_weight)
                        else:
                            current_weights[stock] = min(long_weight_total[stock], self.max_weight)
                    
                    for stock in short_stocks:
                        if self.weight_method == 'equal':
                            current_weights[stock] = max(short_weight, -self.max_weight)
                        else:
                            current_weights[stock] = max(-short_weight_total[stock], -self.max_weight)
            
            weights.loc[date] = current_weights
        
        return weights
    
    def backtest(
        self, 
        weights: pd.DataFrame, 
        returns: pd.DataFrame,
        transaction_cost: float = 0.001
    ) -> pd.DataFrame:
        """
        Backtest portfolio weights against returns.
        
        Args:
            weights: Portfolio weights (dates x symbols)
            returns: Asset returns (dates x symbols)
            transaction_cost: Round-trip transaction cost
            
        Returns:
            DataFrame with portfolio performance
        """
        # Align weights and returns
        common_dates = weights.index.intersection(returns.index)
        common_cols = weights.columns.intersection(returns.columns)
        
        w = weights.loc[common_dates, common_cols]
        r = returns.loc[common_dates, common_cols]
        
        # Calculate portfolio returns
        port_returns = (w.shift(1) * r).sum(axis=1)
        
        # Calculate turnover
        turnover = w.diff().abs().sum(axis=1)
        
        # Subtract transaction costs
        port_returns_net = port_returns - turnover * transaction_cost
        
        # Calculate cumulative returns
        cum_returns = (1 + port_returns_net).cumprod()
        
        # Calculate metrics
        total_return = cum_returns.iloc[-1] - 1
        ann_return = (1 + total_return) ** (252 / len(cum_returns)) - 1
        ann_vol = port_returns_net.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        # Max drawdown
        rolling_max = cum_returns.cummax()
        drawdown = (cum_returns - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        result = pd.DataFrame({
            'daily_return': port_returns_net,
            'cumulative_return': cum_returns,
            'turnover': turnover,
            'drawdown': drawdown
        })
        
        result.attrs['total_return'] = total_return
        result.attrs['ann_return'] = ann_return
        result.attrs['ann_vol'] = ann_vol
        result.attrs['sharpe'] = sharpe
        result.attrs['max_drawdown'] = max_dd
        result.attrs['avg_turnover'] = turnover.mean()
        
        return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_standard_alpha_set() -> List[BaseAlpha]:
    """Create a standard set of alpha factors."""
    return [
        MomentumAlpha(lookback=20, name="mom_20d"),
        MomentumAlpha(lookback=60, name="mom_60d"),
        MomentumAlpha(lookback=120, name="mom_120d"),
        ReversalAlpha(lookback=5, name="reversal_5d"),
        VolatilityAlpha(lookback=20, name="low_vol_20d"),
        VolumeAlpha(lookback=20, name="vol_ratio_20d"),
    ]


def quick_alpha_analysis(
    prices: pd.DataFrame,
    alphas: Optional[List[BaseAlpha]] = None,
    forward_period: int = 5
) -> Tuple[AlphaCombiner, pd.DataFrame]:
    """
    Quick analysis of alpha factors.
    
    Args:
        prices: Price matrix (dates x symbols) or DataFrame with OHLCV
        alphas: List of alpha factors (None = use standard set)
        forward_period: Forward return period for IC calculation
        
    Returns:
        Tuple of (combiner, composite_alpha)
    """
    if alphas is None:
        alphas = create_standard_alpha_set()
    
    # Prepare data
    if 'close' in prices.columns:
        close = prices.pivot_table(index='date', columns='symbol', values='close')
    else:
        close = prices
    
    data = {'prices': close}
    forward_returns = close.pct_change(forward_period).shift(-forward_period)
    
    # Create combiner
    combiner = AlphaCombiner()
    for alpha in alphas:
        combiner.add_alpha(alpha)
    
    # Compute and combine
    combiner.compute_all(data, forward_returns)
    composite = combiner.combine(data, forward_returns, method=CombineMethod.IC_WEIGHTED)
    
    # Print summary
    print("\n=== Alpha Factor Summary ===")
    print(combiner.get_summary().to_string())
    
    return combiner, composite


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: Create synthetic data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='B')
    symbols = ['VNM', 'VIC', 'VHM', 'VCB', 'BID', 'HPG', 'MSN', 'MWG', 'VPB', 'ACB']
    
    # Generate random prices
    prices = pd.DataFrame(
        np.random.randn(len(dates), len(symbols)).cumsum(axis=0) + 100,
        index=dates,
        columns=symbols
    )
    prices = prices.clip(lower=1)  # Ensure positive prices
    
    # Quick analysis
    combiner, composite = quick_alpha_analysis(prices)
    
    # Build portfolio
    portfolio = AlphaPortfolio(n_long=5, n_short=0, rebalance_freq='weekly')
    weights = portfolio.construct(composite)
    
    # Backtest
    returns = prices.pct_change()
    result = portfolio.backtest(weights, returns)
    
    print(f"\n=== Portfolio Performance ===")
    print(f"Total Return: {result.attrs['total_return']:.2%}")
    print(f"Annual Return: {result.attrs['ann_return']:.2%}")
    print(f"Annual Volatility: {result.attrs['ann_vol']:.2%}")
    print(f"Sharpe Ratio: {result.attrs['sharpe']:.2f}")
    print(f"Max Drawdown: {result.attrs['max_drawdown']:.2%}")
    print(f"Avg Turnover: {result.attrs['avg_turnover']:.2%}")
