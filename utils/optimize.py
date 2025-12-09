"""Strategy Optimization Module.

This module provides comprehensive tools for optimizing trading strategy parameters:

1. Train/Validation/Test Split - Proper time-series data splitting
2. Parameter Optimization with Constraints - Optuna-based optimization with trade limits
3. Walk-Forward Validation - Rolling window out-of-sample testing
4. Universal Optimization - Cross-stock parameter optimization
5. Robustness Analysis - Parameter stability and sensitivity testing

Key Features:
    - Minimum/Maximum trade constraints for real-world trading
    - Train/Val/Test splits respecting time-series structure
    - Walk-forward validation for more realistic performance estimates
    - Multiple objective functions (Sharpe, Return, Calmar, etc.)
"""

from __future__ import annotations
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, Tuple, List

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import optuna

warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_TRAIN_RATIO = 0.6
DEFAULT_VAL_RATIO = 0.2
DEFAULT_TEST_RATIO = 0.2
DEFAULT_DATE_COLUMN = 'date'
DEFAULT_TRADING_DAYS_PER_YEAR = 252
DEFAULT_TRAIN_SIZE = 252  # ~1 year
DEFAULT_TEST_SIZE = 63    # ~3 months

# Optuna setup
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None  # type: ignore


# =============================================================================
# DATA SPLITTING FUNCTIONS
# =============================================================================

def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    date_column: str = DEFAULT_DATE_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split time-series data into train, validation, and test sets.

    This is a temporal split - no shuffling to preserve time-series structure.

    Args:
        df: DataFrame with time-series data.
        train_ratio: Proportion for training (default 60%).
        val_ratio: Proportion for validation (default 20%).
        test_ratio: Proportion for testing (default 20%).
        date_column: Name of date column.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        ValueError: If ratios don't sum to 1.0.

    Example:
        >>> train, val, test = train_val_test_split(df, 0.6, 0.2, 0.2)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) >= 0.01:
        raise ValueError("Ratios must sum to 1.0")

    # Sort by date to ensure temporal ordering
    df_sorted = df.sort_values(date_column).reset_index(drop=True)
    n = len(df_sorted)

    # Calculate split indices
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return (
        df_sorted.iloc[:train_end].copy(),
        df_sorted.iloc[train_end:val_end].copy(),
        df_sorted.iloc[val_end:].copy(),
    )


def split_by_date(
    df: pd.DataFrame,
    train_end_date: str,
    val_end_date: str | None = None,
    date_column: str = DEFAULT_DATE_COLUMN,
) -> tuple[pd.DataFrame, ...] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data by specific dates.

    Args:
        df: DataFrame with time-series data.
        train_end_date: End date for training data (format: 'YYYY-MM-DD').
        val_end_date: End date for validation data (optional).
        date_column: Name of date column.

    Returns:
        If val_end_date is None: (train_df, test_df).
        If val_end_date provided: (train_df, val_df, test_df).

    Example:
        >>> train, val, test = split_by_date(df, '2022-01-01', '2023-01-01')
    """
    df_copy = df.copy()
    df_copy[date_column] = pd.to_datetime(df_copy[date_column])

    train_end = pd.to_datetime(train_end_date)

    if val_end_date is None:
        return (
            df_copy[df_copy[date_column] <= train_end].copy(),
            df_copy[df_copy[date_column] > train_end].copy(),
        )

    val_end = pd.to_datetime(val_end_date)
    return (
        df_copy[df_copy[date_column] <= train_end].copy(),
        df_copy[(df_copy[date_column] > train_end) & (df_copy[date_column] <= val_end)].copy(),
        df_copy[df_copy[date_column] > val_end].copy(),
    )


def print_split_info(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    date_column: str = DEFAULT_DATE_COLUMN,
) -> None:
    """Print information about the data splits.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame (optional).
        test_df: Test DataFrame (optional).
        date_column: Name of date column.
    """
    separator = "=" * 60
    print(separator)
    print("DATA SPLIT SUMMARY")
    print(separator)

    print(f"\n📊 Training Set:")
    print(f"   Rows: {len(train_df)}")
    print(f"   Date Range: {train_df[date_column].min()} to {train_df[date_column].max()}")

    if val_df is not None:
        print(f"\n📈 Validation Set:")
        print(f"   Rows: {len(val_df)}")
        print(f"   Date Range: {val_df[date_column].min()} to {val_df[date_column].max()}")

    if test_df is not None:
        print(f"\n🎯 Test Set:")
        print(f"   Rows: {len(test_df)}")
        print(f"   Date Range: {test_df[date_column].min()} to {test_df[date_column].max()}")

    print(separator)


# =============================================================================
# OBJECTIVE FUNCTIONS
# =============================================================================


def sharpe_objective(result: dict[str, Any]) -> float:
    return result['sharpe_ratio']


def calmar_objective(result: dict[str, Any]) -> float:
    if result['max_drawdown'] == 0:
        return 0.0
    return result['annual_return'] / abs(result['max_drawdown'])


def sortino_objective(result: dict[str, Any]) -> float:
    returns = result['data']['strategy_return_net'].dropna()
    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0:
        return result['sharpe_ratio']

    downside_std = downside_returns.std() * np.sqrt(DEFAULT_TRADING_DAYS_PER_YEAR)
    if downside_std == 0:
        return result['sharpe_ratio']

    return result['annual_return'] / downside_std


def combined_objective(
    result: dict[str, Any],
    sharpe_weight: float = 0.4,
    return_weight: float = 0.3,
    drawdown_weight: float = 0.3,
) -> float:
    """Combined objective function considering multiple metrics.

    Score = sharpe_weight * normalized_sharpe
          + return_weight * normalized_return
          - drawdown_weight * normalized_drawdown

    Args:
        result: Backtest result dictionary.
        sharpe_weight: Weight for Sharpe ratio component.
        return_weight: Weight for return component.
        drawdown_weight: Weight for drawdown penalty.

    Returns:
        Combined objective score.
    """
    sharpe = max(result['sharpe_ratio'], -3)  # Clip extreme negative
    ret = result['total_return']
    dd = abs(result['max_drawdown'])

    # Normalize (rough scaling)
    norm_sharpe = sharpe / 2  # Assume good Sharpe ~ 2
    norm_return = ret  # Already in percentage
    norm_dd = dd  # Already in percentage
    
    return (sharpe_weight * norm_sharpe + 
            return_weight * norm_return - 
            drawdown_weight * norm_dd)


# =============================================================================
# PARAMETER OPTIMIZATION WITH VALIDATION
# =============================================================================

def optimize_onestock(train_df: pd.DataFrame,
                             val_df: pd.DataFrame,
                             strategy_class: type,
                             param_space: Dict[str, Tuple],
                             backtest_fn: Callable,
                             n_trials: int = 100,
                             objective_fn: Callable = sharpe_objective,
                             min_trades: int = 10,
                             max_trades: int = None,
                             overfit_penalty: float = 0.3,
                             early_stopping_rounds: int = 20,
                             n_jobs: int = 1,
                             use_pruning: bool = False,
                             timeout: int = None) -> Dict:
    """
    Optimize on training data, validate on validation data to prevent overfitting.
    
    Args:
        train_df: Training data (60% of data typically)
        val_df: Validation data (20% of data typically)
        strategy_class: Strategy class to optimize
        param_space: Parameter search space
            Format: {'param_name': ('type', min, max)}
            Types: 'int', 'float', 'categorical'
        backtest_fn: Backtest function
        n_trials: Number of optimization trials
        objective_fn: Objective function (default: Sharpe ratio)
        min_trades: Minimum trades constraint
        max_trades: Maximum trades constraint
        overfit_penalty: Penalty when train >> val (default 0.3)
        early_stopping_rounds: Stop if no improvement for N trials
        n_jobs: Parallel jobs (-1 for all CPU cores)
        use_pruning: Enable pruning for early stopping of bad trials
        timeout: Max seconds to run
        
    Returns:
        Dictionary with:
            - best_params: Optimal parameters
            - train_score: Score on training data
            - val_score: Score on validation data
            - overfit_gap: train_score - val_score
            - study: Optuna study object
            - all_results: List of all trial results
            
    Example:
        train_df, val_df, test_df = train_val_test_split(df)
        result = optimize_with_validation(
            train_df, val_df, PriceMomentum, param_space, backtest_strategy,
            n_trials=100, n_jobs=-1, use_pruning=True
        )
        # Then evaluate on test_df
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required. Install with: pip install optuna")
    
    all_results = []
    
    # Select pruner
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5) if use_pruning else optuna.pruners.NopPruner()
    
    def objective(trial):
        # Sample parameters
        params = {}
        for param_name, (param_type, *bounds) in param_space.items():
            if param_type == 'int':
                params[param_name] = trial.suggest_int(param_name, bounds[0], bounds[1])
            elif param_type == 'float':
                params[param_name] = trial.suggest_float(param_name, bounds[0], bounds[1])
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, bounds[0])
        
        try:
            # Backtest on train
            strategy_train = strategy_class(**params)
            result_train = backtest_fn(train_df, strategy_train)
            
            # Apply constraints on training
            if result_train['num_trades'] < min_trades:
                return float('-inf')
            if max_trades and result_train['num_trades'] > max_trades:
                return float('-inf')
            
            train_score = objective_fn(result_train)
            
            # Early pruning based on train score
            if use_pruning:
                trial.report(train_score, step=0)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Backtest on validation
            strategy_val = strategy_class(**params)
            result_val = backtest_fn(val_df, strategy_val)
            val_score = objective_fn(result_val)
            
            # Penalize overfitting (train >> val)
            overfit_gap = max(0, train_score - val_score)
            adjusted_score = train_score - overfit_penalty * overfit_gap
            
            # Store results
            all_results.append({
                'params': params.copy(),
                'train_score': train_score,
                'val_score': val_score,
                'overfit_gap': overfit_gap,
                'adjusted_score': adjusted_score,
                'train_trades': result_train['num_trades'],
                'val_trades': result_val['num_trades']
            })
            
            return adjusted_score
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            return float('-inf')
    
    # Early stopping callback
    callbacks = []
    if early_stopping_rounds:
        class EarlyStopper:
            def __init__(self, patience):
                self.patience, self.best, self.count = patience, float('-inf'), 0
            def __call__(self, study, trial):
                if study.best_value > self.best:
                    self.best, self.count = study.best_value, 0
                else:
                    self.count += 1
                    if self.count >= self.patience:
                        study.stop()
        callbacks.append(EarlyStopper(early_stopping_rounds))
    
    # Create study
    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(
        objective, 
        n_trials=n_trials, 
        n_jobs=n_jobs,
        timeout=timeout,
        callbacks=callbacks,
        show_progress_bar=True
    )
    
    # Calculate final scores with best params
    best_params = study.best_params
    result_train = backtest_fn(train_df, strategy_class(**best_params))
    result_val = backtest_fn(val_df, strategy_class(**best_params))
    train_score, val_score = objective_fn(result_train), objective_fn(result_val)
    
    return {
        'best_params': best_params,
        'train_score': train_score,
        'val_score': val_score,
        'overfit_gap': train_score - val_score,
        'train_trades': result_train['num_trades'],
        'val_trades': result_val['num_trades'],
        'study': study,
        'all_results': all_results
    }


# =============================================================================
# WALK-FORWARD OPTIMIZATION WITH VALIDATION
# =============================================================================

def walk_forward_optimization(df: pd.DataFrame,
                               strategy_class: type,
                               param_space: Dict[str, Tuple],
                               backtest_fn: Callable,
                               n_splits: int = 5,
                               train_ratio: float = 0.6,
                               val_ratio: float = 0.2,
                               test_ratio: float = 0.2,
                               n_trials: int = 50,
                               objective_fn: Callable = sharpe_objective,
                               min_trades: int = 5,
                               max_trades: int = None,
                               overfit_penalty: float = 0.3,
                               n_jobs: int = 1,
                               use_pruning: bool = False) -> Dict:
    """
    Walk-forward optimization with Train/Val/Test splits for each window.
    
    For each split:
    1. Split window into train/val/test
    2. Optimize parameters on train, validate on val
    3. Test on out-of-sample test period
    4. Roll forward and repeat
    
    This provides the most realistic estimate of strategy performance.
    
    Args:
        df: Full DataFrame with price data
        strategy_class: Strategy class to optimize
        param_space: Parameter search space
        backtest_fn: Backtest function
        n_splits: Number of walk-forward splits
        train_ratio: Ratio of train within each window (default 0.7)
        val_ratio: Ratio of validation within each window (default 0.15)
        test_ratio: Ratio of test within each window (default 0.15)
        n_trials: Optimization trials per split
        objective_fn: Objective function
        min_trades: Minimum trades constraint
        max_trades: Maximum trades constraint
        overfit_penalty: Penalty for overfitting
        n_jobs: Parallel jobs (-1 for all cores)
        use_pruning: Enable pruning
        
    Returns:
        Dictionary with:
            - splits_results: List of results for each split
            - avg_train_score: Average training score
            - avg_val_score: Average validation score
            - avg_test_score: Average out-of-sample test score
            - avg_overfit_gap: Average (train - test) gap
            - combined_test_returns: Concatenated test period returns
    """
    df = df.sort_values('date').reset_index(drop=True)
    n = len(df)
    
    # Calculate window size for each split (with overlap)
    window_size = n // n_splits
    step_size = (n - window_size) // (n_splits - 1) if n_splits > 1 else 0
    
    splits_results = []
    all_test_returns = []
    
    print(f"\n{'='*70}")
    print(f"WALK-FORWARD OPTIMIZATION WITH VALIDATION ({n_splits} splits)")
    print(f"{'='*70}")
    print(f"Window: Train({train_ratio:.0%}) / Val({val_ratio:.0%}) / Test({test_ratio:.0%})")
    
    for i in range(n_splits):
        # Define window boundaries
        start_idx = i * step_size
        end_idx = min(start_idx + window_size, n)
        
        window_df = df.iloc[start_idx:end_idx].copy()
        window_n = len(window_df)
        
        # Split window into train/val/test
        train_end = int(window_n * train_ratio)
        val_end = int(window_n * (train_ratio + val_ratio))
        
        train_df = window_df.iloc[:train_end].copy()
        val_df = window_df.iloc[train_end:val_end].copy()
        test_df = window_df.iloc[val_end:].copy()
        
        print(f"\n📊 Split {i+1}/{n_splits}")
        print(f"   Train: {train_df['date'].min()} to {train_df['date'].max()} ({len(train_df)} days)")
        print(f"   Val:   {val_df['date'].min()} to {val_df['date'].max()} ({len(val_df)} days)")
        print(f"   Test:  {test_df['date'].min()} to {test_df['date'].max()} ({len(test_df)} days)")
        
        # Optimize on train, validate on val
        opt_result = optimize_onestock(
            train_df, val_df, 
            strategy_class, param_space, backtest_fn,
            n_trials=n_trials, 
            objective_fn=objective_fn,
            min_trades=min_trades, 
            max_trades=max_trades,
            overfit_penalty=overfit_penalty,
            n_jobs=n_jobs,
            use_pruning=use_pruning,
            early_stopping_rounds=15
        )
        
        best_params = opt_result['best_params']
        train_score = opt_result['train_score']
        val_score = opt_result['val_score']
        
        # Test on out-of-sample test period
        strategy_test = strategy_class(**best_params)
        result_test = backtest_fn(test_df, strategy_test)
        test_score = objective_fn(result_test)
        
        print(f"   Best params: {best_params}")
        print(f"   Train: {train_score:.3f} | Val: {val_score:.3f} | Test: {test_score:.3f}")
        print(f"   Overfit Gap (Train-Test): {train_score - test_score:.3f}")
        print(f"   Test Return: {result_test['total_return']:.2%}")
        
        splits_results.append({
            'split': i + 1,
            'train_period': f"{train_df['date'].min()} to {train_df['date'].max()}",
            'val_period': f"{val_df['date'].min()} to {val_df['date'].max()}",
            'test_period': f"{test_df['date'].min()} to {test_df['date'].max()}",
            'best_params': best_params,
            'train_score': train_score,
            'val_score': val_score,
            'test_score': test_score,
            'overfit_gap': train_score - test_score,
            'test_return': result_test['total_return'],
            'test_trades': result_test['num_trades']
        })
        
        # Collect test returns for combined analysis
        test_returns = result_test['data']['strategy_return_net'].dropna()
        all_test_returns.extend(test_returns.tolist())
    
    # Calculate summary statistics
    train_scores, val_scores, test_scores = [], [], []
    overfit_gaps, test_returns_list = [], []
    for r in splits_results:
        train_scores.append(r['train_score'])
        val_scores.append(r['val_score'])
        test_scores.append(r['test_score'])
        overfit_gaps.append(r['overfit_gap'])
        test_returns_list.append(r['test_return'])
    
    # Combined test period performance
    combined_returns = pd.Series(all_test_returns)
    combined_sharpe = (combined_returns.mean() / combined_returns.std() * np.sqrt(252)) if combined_returns.std() > 0 else 0
    combined_total_return = (1 + combined_returns).prod() - 1
    
    summary = {
        'splits_results': splits_results,
        'avg_train_score': np.mean(train_scores),
        'avg_val_score': np.mean(val_scores),
        'avg_test_score': np.mean(test_scores),
        'std_test_score': np.std(test_scores),
        'min_test_score': np.min(test_scores),
        'max_test_score': np.max(test_scores),
        'avg_overfit_gap': np.mean(overfit_gaps),
        'avg_test_return': np.mean(test_returns_list),
        'combined_sharpe': combined_sharpe,
        'combined_return': combined_total_return
    }
    
    print(f"\n{'='*70}")
    print("WALK-FORWARD SUMMARY")
    print(f"{'='*70}")
    print(f"Average Train Sharpe: {summary['avg_train_score']:.3f}")
    print(f"Average Val Sharpe:   {summary['avg_val_score']:.3f}")
    print(f"Average Test Sharpe:  {summary['avg_test_score']:.3f} (±{summary['std_test_score']:.3f})")
    print(f"Average Overfit Gap:  {summary['avg_overfit_gap']:.3f}")
    print(f"Combined Test Return: {summary['combined_return']:.2%}")
    print(f"Combined Test Sharpe: {summary['combined_sharpe']:.3f}")
    
    # Overfitting analysis
    if summary['avg_overfit_gap'] < 0.3:
        print("Good generalization (low overfit gap)")
    elif summary['avg_overfit_gap'] < 0.6:
        print("Moderate overfitting")
    else:
        print("Significant overfitting detected")
    
    return summary


# =============================================================================
# UNIVERSAL (CROSS-STOCK) OPTIMIZATION WITH VALIDATION
# =============================================================================

def optimize_universal(all_train_data: Dict[str, pd.DataFrame],
                       all_val_data: Dict[str, pd.DataFrame],
                       strategy_class: type,
                       param_space: Dict[str, Tuple],
                       backtest_fn: Callable,
                       n_trials: int = 100,
                       objective_fn: Callable = sharpe_objective,
                       min_trades: int = 10,
                       max_trades: int = None,
                       consistency_penalty: float = 0.1,
                       overfit_penalty: float = 0.3,
                       n_jobs: int = 1,
                       use_pruning: bool = False,
                       timeout: int = None,
                       early_stop_patience: int = None) -> Dict:
    """
    Optimize strategy parameters across multiple stocks with validation.
    
    This produces robust parameters that:
    1. Work well on average across different stocks (generalization)
    2. Don't overfit to training data (train/val split per stock)
    
    Args:
        all_train_data: Dictionary of {symbol: train_DataFrame}
        all_val_data: Dictionary of {symbol: val_DataFrame}
        strategy_class: Strategy class to optimize
        param_space: Parameter search space
        backtest_fn: Backtest function
        n_trials: Number of optimization trials
        objective_fn: Objective function
        min_trades: Minimum trades per stock
        max_trades: Maximum trades per stock
        consistency_penalty: Penalty for high variance across stocks (0-1)
        overfit_penalty: Penalty when train >> val (default 0.3)
        n_jobs: Parallel jobs (-1 for all CPU cores)
        use_pruning: Enable pruning for early stopping of bad trials
        timeout: Max seconds to run
        early_stop_patience: Stop if no improvement for N trials
        
    Returns:
        Dictionary with:
            - best_params: Optimal parameters
            - avg_train_score: Average training score across stocks
            - avg_val_score: Average validation score across stocks
            - avg_overfit_gap: Average (train - val) gap
            - per_stock_results: Dict of results per stock
            - study: Optuna study object
        
    Example:
        # Split data for all stocks
        all_train, all_val, all_test = {}, {}, {}
        for symbol, df in all_data.items():
            train, val, test = train_val_test_split(df)
            all_train[symbol] = train
            all_val[symbol] = val
            all_test[symbol] = test
        
        # Optimize
        result = optimize_universal(
            all_train, all_val, PriceMomentum, param_space, backtest_strategy,
            n_trials=100, n_jobs=-1, use_pruning=True
        )
        
        # Evaluate on test
        for symbol, test_df in all_test.items():
            strategy = PriceMomentum(**result['best_params'])
            test_result = backtest_strategy(test_df, strategy)
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required. Install with: pip install optuna")
    
    symbols = list(all_train_data.keys())
    n_stocks = len(symbols)
    
    # Select pruner
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5) if use_pruning else optuna.pruners.NopPruner()
    
    def objective(trial):
        # Sample parameters
        params = {}
        for param_name, (param_type, *bounds) in param_space.items():
            if param_type == 'int':
                params[param_name] = trial.suggest_int(param_name, bounds[0], bounds[1])
            elif param_type == 'float':
                params[param_name] = trial.suggest_float(param_name, bounds[0], bounds[1])
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, bounds[0])
        
        train_scores = []
        val_scores = []
        valid_stocks = 0
        
        # Early pruning: test on subset of stocks first
        if use_pruning:
            early_scores = []
            for symbol in symbols[:min(2, n_stocks)]:
                try:
                    strategy = strategy_class(**params)
                    result = backtest_fn(all_train_data[symbol], strategy)
                    if result['num_trades'] >= min_trades:
                        early_scores.append(objective_fn(result))
                except:
                    pass
            
            if len(early_scores) > 0:
                early_score = np.mean(early_scores)
                trial.report(early_score, step=0)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        # Full evaluation on all stocks (train + val)
        for symbol in symbols:
            try:
                # Train score
                strategy_train = strategy_class(**params)
                result_train = backtest_fn(all_train_data[symbol], strategy_train)
                
                # Check constraints
                if result_train['num_trades'] < min_trades:
                    continue
                if max_trades and result_train['num_trades'] > max_trades:
                    continue
                
                train_score = objective_fn(result_train)
                
                # Validation score
                strategy_val = strategy_class(**params)
                result_val = backtest_fn(all_val_data[symbol], strategy_val)
                val_score = objective_fn(result_val)
                
                train_scores.append(train_score)
                val_scores.append(val_score)
                valid_stocks += 1
                
            except optuna.TrialPruned:
                raise
            except Exception:
                continue
        
        # Require at least half the stocks to pass constraints
        if valid_stocks < n_stocks / 2:
            return float('-inf')
        
        # Calculate metrics
        avg_train = np.mean(train_scores)
        avg_val = np.mean(val_scores)
        std_val = np.std(val_scores)
        
        # Penalize overfitting (train >> val)
        overfit_gap = max(0, avg_train - avg_val)
        
        # Final score: high val score, low variance, low overfit
        final_score = avg_val - consistency_penalty * std_val - overfit_penalty * overfit_gap
        
        if use_pruning:
            trial.report(final_score, step=1)
        
        return final_score
    
    # Early stopping callback
    callbacks = []
    if early_stop_patience:
        class EarlyStoppingCallback:
            def __init__(self, patience: int):
                self.patience = patience
                self.best_value = float('-inf')
                self.no_improve_count = 0
                
            def __call__(self, study, trial):
                if study.best_value > self.best_value:
                    self.best_value = study.best_value
                    self.no_improve_count = 0
                else:
                    self.no_improve_count += 1
                    
                if self.no_improve_count >= self.patience:
                    study.stop()
        
        callbacks.append(EarlyStoppingCallback(early_stop_patience))
    
    # Create study
    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(
        objective, 
        n_trials=n_trials, 
        n_jobs=n_jobs,
        timeout=timeout,
        callbacks=callbacks,
        show_progress_bar=True
    )
    
    # Calculate final scores per stock with best params
    best_params = study.best_params
    per_stock_results, train_scores, val_scores = {}, [], []
    
    for symbol in symbols:
        try:
            result_train = backtest_fn(all_train_data[symbol], strategy_class(**best_params))
            result_val = backtest_fn(all_val_data[symbol], strategy_class(**best_params))
            train_score, val_score = objective_fn(result_train), objective_fn(result_val)
            
            per_stock_results[symbol] = {
                'train_score': train_score,
                'val_score': val_score,
                'overfit_gap': train_score - val_score,
                'train_trades': result_train['num_trades'],
                'val_trades': result_val['num_trades']
            }
            train_scores.append(train_score)
            val_scores.append(val_score)
        except:
            per_stock_results[symbol] = {'error': 'Failed'}
    
    return {
        'best_params': best_params,
        'avg_train_score': np.mean(train_scores) if train_scores else np.nan,
        'avg_val_score': np.mean(val_scores) if val_scores else np.nan,
        'std_val_score': np.std(val_scores) if val_scores else np.nan,
        'avg_overfit_gap': np.mean([r['overfit_gap'] for r in per_stock_results.values() if 'overfit_gap' in r]),
        'per_stock_results': per_stock_results,
        'study': study
    }


# =============================================================================
# ROBUSTNESS ANALYSIS
# =============================================================================

def bootstrap_performance(df: pd.DataFrame,
                          strategy_class: type,
                          params: Dict,
                          backtest_fn: Callable,
                          n_bootstrap: int = 100,
                          sample_ratio: float = 0.8) -> Dict:
    """
    Bootstrap analysis to estimate confidence intervals of performance.
    
    Args:
        df: DataFrame with price data
        strategy_class: Strategy class
        params: Strategy parameters
        n_bootstrap: Number of bootstrap samples
        sample_ratio: Ratio of data to sample each iteration
        
    Returns:
        Dictionary with mean, std, and confidence intervals
    """
    sharpes = []
    returns = []
    
    n = len(df)
    sample_size = int(n * sample_ratio)
    
    for _ in range(n_bootstrap):
        indices = np.sort(np.random.choice(n, sample_size, replace=True))
        sample_df = df.iloc[indices].reset_index(drop=True)
        try:
            result = backtest_fn(sample_df, strategy_class(**params))
            sharpes.append(result['sharpe_ratio'])
            returns.append(result['total_return'])
        except:
            continue
    
    return {
        'sharpe_mean': np.mean(sharpes),
        'sharpe_std': np.std(sharpes),
        'sharpe_ci_95': (np.percentile(sharpes, 2.5), np.percentile(sharpes, 97.5)),
        'return_mean': np.mean(returns),
        'return_std': np.std(returns),
        'return_ci_95': (np.percentile(returns, 2.5), np.percentile(returns, 97.5)),
        'n_successful': len(sharpes)
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data splitting
    'train_val_test_split',
    'split_by_date',
    'print_split_info',
    
    # Objective functions
    'sharpe_objective',
    'calmar_objective',
    'sortino_objective',
    'combined_objective',
    
    # Optimization with Validation
    'optimize_onestock',
    'optimize_universal',
    'walk_forward_optimization',
    
    # Robustness
    'bootstrap_performance',
]
