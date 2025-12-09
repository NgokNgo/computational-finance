"""Strategy Optimization Module - Optimized Version.

Comprehensive tools for optimizing trading strategy parameters with proper validation.
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

DEFAULT_TRAIN_RATIO, DEFAULT_VAL_RATIO, DEFAULT_TEST_RATIO = 0.6, 0.2, 0.2
DEFAULT_DATE_COLUMN = 'date'
DEFAULT_TRADING_DAYS_PER_YEAR = 252
DEFAULT_TRAIN_SIZE, DEFAULT_TEST_SIZE = 252, 63

# Optuna setup
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE, optuna = False, None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _sample_params(trial, param_space: Dict[str, Tuple]) -> Dict:
    """Sample parameters from parameter space."""
    params = {}
    for name, (ptype, *bounds) in param_space.items():
        if ptype == 'int':
            params[name] = trial.suggest_int(name, bounds[0], bounds[1])
        elif ptype == 'float':
            params[name] = trial.suggest_float(name, bounds[0], bounds[1])
        elif ptype == 'categorical':
            params[name] = trial.suggest_categorical(name, bounds[0])
    return params


def _create_early_stopper(patience: int):
    """Create early stopping callback."""
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
    return EarlyStopper(patience)


# =============================================================================
# DATA SPLITTING FUNCTIONS
# =============================================================================

def train_val_test_split(df: pd.DataFrame,
                        train_ratio: float = DEFAULT_TRAIN_RATIO,
                        val_ratio: float = DEFAULT_VAL_RATIO,
                        test_ratio: float = DEFAULT_TEST_RATIO,
                        date_column: str = DEFAULT_DATE_COLUMN,
                        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split time-series data into train, validation, and test sets."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) >= 0.01:
        raise ValueError("Ratios must sum to 1.0")

    df_sorted = df.sort_values(date_column).reset_index(drop=True)
    n = len(df_sorted)
    train_end, val_end = int(n * train_ratio), int(n * (train_ratio + val_ratio))

    return (
        df_sorted.iloc[:train_end].copy(),
        df_sorted.iloc[train_end:val_end].copy(),
        df_sorted.iloc[val_end:].copy(),
    )


def split_by_date(df: pd.DataFrame,
                train_end_date: str,
                val_end_date: str | None = None,
                date_column: str = DEFAULT_DATE_COLUMN,
                ) -> tuple[pd.DataFrame, ...]:
    """Split data by specific dates."""
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


def print_split_info(train_df: pd.DataFrame,
                    val_df: pd.DataFrame | None = None,
                    test_df: pd.DataFrame | None = None,
                    date_column: str = DEFAULT_DATE_COLUMN,
                    ) -> None:
    """Print information about the data splits."""
    print("=" * 60)
    print("DATA SPLIT SUMMARY")
    print("=" * 60)
    print(f"\n📊 Training Set: {len(train_df)} rows")
    print(f"   {train_df[date_column].min()} to {train_df[date_column].max()}")
    
    if val_df is not None:
        print(f"\n📈 Validation Set: {len(val_df)} rows")
        print(f"   {val_df[date_column].min()} to {val_df[date_column].max()}")
    
    if test_df is not None:
        print(f"\n🎯 Test Set: {len(test_df)} rows")
        print(f"   {test_df[date_column].min()} to {test_df[date_column].max()}")
    print("=" * 60)


# =============================================================================
# OBJECTIVE FUNCTIONS
# =============================================================================

def sharpe_objective(result: dict[str, Any]) -> float:
    return result['sharpe_ratio']


def calmar_objective(result: dict[str, Any]) -> float:
    return result['annual_return'] / abs(result['max_drawdown']) if result['max_drawdown'] != 0 else 0.0


def sortino_objective(result: dict[str, Any]) -> float:
    returns = result['data']['strategy_return_net'].dropna()
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        return result['sharpe_ratio']
    
    downside_std = downside_returns.std() * np.sqrt(DEFAULT_TRADING_DAYS_PER_YEAR)
    return result['annual_return'] / downside_std if downside_std != 0 else result['sharpe_ratio']


def combined_objective(result: dict[str, Any],
                    sharpe_weight: float = 0.4,
                    return_weight: float = 0.3,
                    drawdown_weight: float = 0.3,
                    ) -> float:
    """Combined objective: Sharpe + Return - Drawdown."""
    sharpe = max(result['sharpe_ratio'], -3)
    norm_sharpe = sharpe / 2
    norm_return = result['total_return']
    norm_dd = abs(result['max_drawdown'])
    return sharpe_weight * norm_sharpe + return_weight * norm_return - drawdown_weight * norm_dd


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
                    timeout: int = None
                    ) -> Dict:
    """Optimize on training, validate on validation data to prevent overfitting."""
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna required: pip install optuna")
    
    all_results = []
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5) if use_pruning else optuna.pruners.NopPruner()
    
    def objective(trial):
        params = _sample_params(trial, param_space)
        try:
            result_train = backtest_fn(train_df, strategy_class(**params))
            
            # Apply constraints
            if result_train['num_trades'] < min_trades:
                return float('-inf')
            if max_trades and result_train['num_trades'] > max_trades:
                return float('-inf')
            
            train_score = objective_fn(result_train)
            
            # Early pruning
            if use_pruning:
                trial.report(train_score, step=0)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Validate
            result_val = backtest_fn(val_df, strategy_class(**params))
            val_score = objective_fn(result_val)
            
            # Penalize overfitting
            overfit_gap = max(0, train_score - val_score)
            adjusted_score = train_score - overfit_penalty * overfit_gap
            
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
        except Exception:
            return float('-inf')
    
    # Setup study
    callbacks = [_create_early_stopper(early_stopping_rounds)] if early_stopping_rounds else []
    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, timeout=timeout, 
                   callbacks=callbacks, show_progress_bar=True)
    
    # Final scores
    best_params = study.best_params
    result_train = backtest_fn(train_df, strategy_class(**best_params))
    result_val = backtest_fn(val_df, strategy_class(**best_params))
    
    return {
        'best_params': best_params,
        'train_score': objective_fn(result_train),
        'val_score': objective_fn(result_val),
        'overfit_gap': objective_fn(result_train) - objective_fn(result_val),
        'train_trades': result_train['num_trades'],
        'val_trades': result_val['num_trades'],
        'study': study,
        'all_results': all_results
    }


# =============================================================================
# WALK-FORWARD OPTIMIZATION
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
                            use_pruning: bool = False
                            ) -> Dict:
    """Walk-forward optimization with Train/Val/Test splits."""
    df = df.sort_values('date').reset_index(drop=True)
    n = len(df)
    window_size = n // n_splits
    step_size = (n - window_size) // (n_splits - 1) if n_splits > 1 else 0
    
    splits_results, all_test_returns = [], []
    
    print(f"\n{'='*70}")
    print(f"WALK-FORWARD OPTIMIZATION ({n_splits} splits)")
    print(f"{'='*70}")
    print(f"Window: Train({train_ratio:.0%}) / Val({val_ratio:.0%}) / Test({test_ratio:.0%})")
    
    for i in range(n_splits):
        start_idx, end_idx = i * step_size, min(i * step_size + window_size, n)
        window_df = df.iloc[start_idx:end_idx].copy()
        window_n = len(window_df)
        
        # Split window
        train_end = int(window_n * train_ratio)
        val_end = int(window_n * (train_ratio + val_ratio))
        train_df = window_df.iloc[:train_end].copy()
        val_df = window_df.iloc[train_end:val_end].copy()
        test_df = window_df.iloc[val_end:].copy()
        
        print(f"\n📊 Split {i+1}/{n_splits}")
        print(f"   Train: {train_df['date'].min()} to {train_df['date'].max()} ({len(train_df)} days)")
        print(f"   Val:   {val_df['date'].min()} to {val_df['date'].max()} ({len(val_df)} days)")
        print(f"   Test:  {test_df['date'].min()} to {test_df['date'].max()} ({len(test_df)} days)")
        
        # Optimize
        opt_result = optimize_onestock(
            train_df, val_df, strategy_class, param_space, backtest_fn,
            n_trials=n_trials, objective_fn=objective_fn, min_trades=min_trades, 
            max_trades=max_trades, overfit_penalty=overfit_penalty, n_jobs=n_jobs,
            use_pruning=use_pruning, early_stopping_rounds=15
        )
        
        # Test
        result_test = backtest_fn(test_df, strategy_class(**opt_result['best_params']))
        test_score = objective_fn(result_test)
        
        print(f"   Params: {opt_result['best_params']}")
        print(f"   Train: {opt_result['train_score']:.3f} | Val: {opt_result['val_score']:.3f} | Test: {test_score:.3f}")
        print(f"   Overfit Gap: {opt_result['train_score'] - test_score:.3f}")
        print(f"   Test Return: {result_test['total_return']:.2%}")
        
        splits_results.append({
            'split': i + 1,
            'train_period': f"{train_df['date'].min()} to {train_df['date'].max()}",
            'val_period': f"{val_df['date'].min()} to {val_df['date'].max()}",
            'test_period': f"{test_df['date'].min()} to {test_df['date'].max()}",
            'best_params': opt_result['best_params'],
            'train_score': opt_result['train_score'],
            'val_score': opt_result['val_score'],
            'test_score': test_score,
            'overfit_gap': opt_result['train_score'] - test_score,
            'test_return': result_test['total_return'],
            'test_trades': result_test['num_trades']
        })
        
        all_test_returns.extend(result_test['data']['strategy_return_net'].dropna().tolist())
    
    # Summary
    train_scores = [r['train_score'] for r in splits_results]
    val_scores = [r['val_score'] for r in splits_results]
    test_scores = [r['test_score'] for r in splits_results]
    overfit_gaps = [r['overfit_gap'] for r in splits_results]
    
    combined_returns = pd.Series(all_test_returns)
    combined_sharpe = (combined_returns.mean() / combined_returns.std() * np.sqrt(252)) if combined_returns.std() > 0 else 0
    
    summary = {
        'splits_results': splits_results,
        'avg_train_score': np.mean(train_scores),
        'avg_val_score': np.mean(val_scores),
        'avg_test_score': np.mean(test_scores),
        'std_test_score': np.std(test_scores),
        'min_test_score': np.min(test_scores),
        'max_test_score': np.max(test_scores),
        'avg_overfit_gap': np.mean(overfit_gaps),
        'avg_test_return': np.mean([r['test_return'] for r in splits_results]),
        'combined_sharpe': combined_sharpe,
        'combined_return': (1 + combined_returns).prod() - 1
    }
    
    print(f"\n{'='*70}")
    print("WALK-FORWARD SUMMARY")
    print(f"{'='*70}")
    print(f"Avg Train Sharpe: {summary['avg_train_score']:.3f}")
    print(f"Avg Val Sharpe:   {summary['avg_val_score']:.3f}")
    print(f"Avg Test Sharpe:  {summary['avg_test_score']:.3f} (±{summary['std_test_score']:.3f})")
    print(f"Avg Overfit Gap:  {summary['avg_overfit_gap']:.3f}")
    print(f"Combined Return:  {summary['combined_return']:.2%}")
    print(f"Combined Sharpe:  {summary['combined_sharpe']:.3f}")
    
    if summary['avg_overfit_gap'] < 0.3:
        print("✅ Good generalization")
    elif summary['avg_overfit_gap'] < 0.6:
        print("⚠️ Moderate overfitting")
    else:
        print("❌ Significant overfitting")
    
    return summary


# =============================================================================
# UNIVERSAL OPTIMIZATION
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
                    early_stop_patience: int = None
                    ) -> Dict:
    """Optimize across multiple stocks with validation."""
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna required: pip install optuna")
    
    symbols = list(all_train_data.keys())
    n_stocks = len(symbols)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5) if use_pruning else optuna.pruners.NopPruner()
    
    def objective(trial):
        params = _sample_params(trial, param_space)
        train_scores, val_scores, valid_stocks = [], [], 0
        
        # Early pruning on subset
        if use_pruning:
            early_scores = []
            for symbol in symbols[:min(2, n_stocks)]:
                try:
                    result = backtest_fn(all_train_data[symbol], strategy_class(**params))
                    if result['num_trades'] >= min_trades:
                        early_scores.append(objective_fn(result))
                except:
                    pass
            
            if len(early_scores) > 0:
                trial.report(np.mean(early_scores), step=0)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        
        # Full evaluation
        for symbol in symbols:
            try:
                result_train = backtest_fn(all_train_data[symbol], strategy_class(**params))
                if result_train['num_trades'] < min_trades:
                    continue
                if max_trades and result_train['num_trades'] > max_trades:
                    continue
                
                result_val = backtest_fn(all_val_data[symbol], strategy_class(**params))
                train_scores.append(objective_fn(result_train))
                val_scores.append(objective_fn(result_val))
                valid_stocks += 1
            except optuna.TrialPruned:
                raise
            except:
                continue
        
        if valid_stocks < n_stocks / 2:
            return float('-inf')
        
        avg_val = np.mean(val_scores)
        std_val = np.std(val_scores)
        overfit_gap = max(0, np.mean(train_scores) - avg_val)
        final_score = avg_val - consistency_penalty * std_val - overfit_penalty * overfit_gap
        
        if use_pruning:
            trial.report(final_score, step=1)
        return final_score
    
    # Setup study
    callbacks = [_create_early_stopper(early_stop_patience)] if early_stop_patience else []
    study = optuna.create_study(direction="maximize", pruner=pruner, 
                                sampler=optuna.samplers.TPESampler(seed=42))
    
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, timeout=timeout,
                   callbacks=callbacks, show_progress_bar=True)
    
    # Final scores per stock
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
                        sample_ratio: float = 0.8
                        ) -> Dict:
    """Bootstrap analysis for confidence intervals."""
    sharpes, returns = [], []
    n, sample_size = len(df), int(len(df) * sample_ratio)
    
    for _ in range(n_bootstrap):
        indices = np.sort(np.random.choice(n, sample_size, replace=True))
        try:
            result = backtest_fn(df.iloc[indices].reset_index(drop=True), strategy_class(**params))
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
    'train_val_test_split', 'split_by_date', 'print_split_info',
    'sharpe_objective', 'calmar_objective', 'sortino_objective', 'combined_objective',
    'optimize_onestock', 'optimize_universal', 'walk_forward_optimization',
    'bootstrap_performance',
]
