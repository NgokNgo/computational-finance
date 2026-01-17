"""
Optimizer Core Module - Lean-inspired Optimizer Architecture

Based on QuantConnect/Lean modular design, providing:
- Event-driven optimization pipeline
- Multiple optimization strategies (Grid, Bayesian, Genetic)
- Comprehensive result analysis and reporting
- Pluggable objective functions and constraints
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class OptimizationStrategy(Enum):
    """Optimization strategy types."""
    GRID_SEARCH = "grid_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    RANDOM = "random"


class ObjectiveDirection(Enum):
    """Direction to optimize towards."""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ConstraintType(Enum):
    """Types of constraints."""
    HARD = "hard"  # Trial rejected if violated
    SOFT = "soft"  # Penalty applied if violated


# =============================================================================
# RESULT MODELS
# =============================================================================

@dataclass
class OptimizationResult:
    """Result of a single optimization trial."""
    trial_id: int
    params: Dict[str, Any]
    score: float
    train_score: Optional[float] = None
    val_score: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    constraints_violated: List[str] = field(default_factory=list)
    error: Optional[str] = None
    
    def is_valid(self) -> bool:
        """Check if result is valid (no errors, constraints passed)."""
        return self.error is None and len(self.constraints_violated) == 0


@dataclass
class OptimizationSummary:
    """Summary of optimization results."""
    best_params: Dict[str, Any]
    best_score: float
    best_trial_id: int
    n_trials: int
    n_valid_trials: int
    direction: ObjectiveDirection
    
    # Statistics
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    
    # Per-stock results (for universe optimization)
    per_stock_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Duration
    elapsed_time: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""
    strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN
    direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE
    n_trials: int = 100
    n_jobs: int = 1
    timeout: Optional[int] = None
    seed: int = 42
    verbose: bool = True
    
    # Early stopping
    early_stopping_patience: Optional[int] = None
    early_stopping_min_delta: float = 1e-4
    
    # Pruning
    use_pruning: bool = False
    pruning_patience: int = 5


# =============================================================================
# CONSTRAINT DEFINITIONS
# =============================================================================

class Constraint(ABC):
    """Base class for constraints."""
    
    def __init__(self, name: str, ctype: ConstraintType):
        self.name = name
        self.type = ctype
    
    @abstractmethod
    def check(self, result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Check if constraint is satisfied.
        
        Returns:
            (is_satisfied, violation_reason)
        """
        pass


class TradeCountConstraint(Constraint):
    """Constraint on number of trades."""
    
    def __init__(self, min_trades: int, max_trades: Optional[int] = None, ctype: ConstraintType = ConstraintType.HARD):
        super().__init__("trade_count", ctype)
        self.min_trades = min_trades
        self.max_trades = max_trades
    
    def check(self, result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        num_trades = result.get('num_trades', 0)
        if num_trades < self.min_trades:
            return False, f"Too few trades: {num_trades} < {self.min_trades}"
        if self.max_trades and num_trades > self.max_trades:
            return False, f"Too many trades: {num_trades} > {self.max_trades}"
        return True, None


class SharpeConstraint(Constraint):
    """Constraint on Sharpe ratio."""
    
    def __init__(self, min_sharpe: float = 0.5, ctype: ConstraintType = ConstraintType.SOFT):
        super().__init__("sharpe", ctype)
        self.min_sharpe = min_sharpe
    
    def check(self, result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        sharpe = result.get('sharpe_ratio', 0)
        if sharpe < self.min_sharpe:
            return False, f"Sharpe below threshold: {sharpe:.3f} < {self.min_sharpe}"
        return True, None


class DrawdownConstraint(Constraint):
    """Constraint on maximum drawdown."""
    
    def __init__(self, max_drawdown: float = -0.2, ctype: ConstraintType = ConstraintType.SOFT):
        super().__init__("drawdown", ctype)
        self.max_drawdown = max_drawdown
    
    def check(self, result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        dd = result.get('max_drawdown', 0)
        if dd < self.max_drawdown:
            return False, f"Drawdown exceeds threshold: {dd:.3f} < {self.max_drawdown}"
        return True, None


class ReturnConstraint(Constraint):
    """Constraint on total return."""
    
    def __init__(self, min_return: float = 0.0, ctype: ConstraintType = ConstraintType.SOFT):
        super().__init__("return", ctype)
        self.min_return = min_return
    
    def check(self, result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        ret = result.get('total_return', 0)
        if ret < self.min_return:
            return False, f"Return below threshold: {ret:.3f} < {self.min_return}"
        return True, None


# =============================================================================
# OBJECTIVE FUNCTION BUILDER
# =============================================================================

class ObjectiveBuilder:
    """Builder for complex objective functions with penalties."""
    
    def __init__(self, direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE):
        self.direction = direction
        self.components: List[Tuple[Callable, float]] = []
        self.penalties: List[Tuple[str, float]] = []
    
    def add_component(self, fn: Callable, weight: float = 1.0) -> ObjectiveBuilder:
        """Add weighted objective component."""
        self.components.append((fn, weight))
        return self
    
    def add_penalty(self, name: str, weight: float = 1.0) -> ObjectiveBuilder:
        """Add penalty component."""
        self.penalties.append((name, weight))
        return self
    
    def build(self) -> Callable:
        """Build the final objective function."""
        
        def objective(result: Dict[str, Any]) -> float:
            # Main components
            score = 0.0
            for fn, weight in self.components:
                score += weight * fn(result)
            
            # Penalties
            for penalty_name, penalty_weight in self.penalties:
                if penalty_name == 'overfit':
                    overfit_gap = max(0, result.get('train_score', 0) - result.get('val_score', 0))
                    score -= penalty_weight * overfit_gap
                elif penalty_name == 'consistency':
                    # Assumes result contains 'val_scores' list for multiple assets
                    val_scores = result.get('val_scores', [result.get('val_score', 0)])
                    consistency = np.std(val_scores) if len(val_scores) > 1 else 0
                    score -= penalty_weight * consistency
                elif penalty_name == 'trade_variance':
                    train_trades = result.get('train_trades', [])
                    if len(train_trades) > 1:
                        trade_cv = np.std(train_trades) / (np.mean(train_trades) + 1e-8)
                        score -= penalty_weight * trade_cv
            
            return score if self.direction == ObjectiveDirection.MAXIMIZE else -score
        
        return objective


# =============================================================================
# OPTIMIZER BASE CLASS
# =============================================================================

class OptimizerBase(ABC):
    """Base class for all optimizers."""
    
    def __init__(self, config: OptimizerConfig):
        self.config = config
        self.trials: List[OptimizationResult] = []
        self.best_result: Optional[OptimizationResult] = None
    
    @abstractmethod
    def optimize(self,
                 train_data: Dict[str, pd.DataFrame] | pd.DataFrame,
                 val_data: Dict[str, pd.DataFrame] | pd.DataFrame,
                 strategy_class: type,
                 param_space: Dict[str, Tuple],
                 backtest_fn: Callable,
                 objective_fn: Callable,
                 constraints: List[Constraint] = None) -> OptimizationSummary:
        """Run optimization."""
        pass
    
    def _check_constraints(self, result: Dict[str, Any], constraints: List[Constraint]) -> Tuple[bool, List[str]]:
        """Check all constraints on a result."""
        if not constraints:
            return True, []
        
        violations = []
        for constraint in constraints:
            satisfied, reason = constraint.check(result)
            if not satisfied:
                violations.append(reason)
                if constraint.type == ConstraintType.HARD:
                    return False, violations
        
        return True, violations
    
    def _record_trial(self, trial_id: int, params: Dict, score: float, 
                     train_score: Optional[float] = None,
                     val_score: Optional[float] = None,
                     metrics: Dict = None,
                     violations: List[str] = None,
                     error: Optional[str] = None):
        """Record optimization trial result."""
        result = OptimizationResult(
            trial_id=trial_id,
            params=params,
            score=score,
            train_score=train_score,
            val_score=val_score,
            metrics=metrics or {},
            constraints_violated=violations or [],
            error=error
        )
        self.trials.append(result)
        
        if error is None and len(violations or []) == 0:
            if self.best_result is None or score > self.best_result.score:
                self.best_result = result


# =============================================================================
# GRID SEARCH OPTIMIZER (Without Optuna)
# =============================================================================

class GridSearchOptimizer(OptimizerBase):
    """Grid search optimization - exhaustive parameter search without external libraries."""
    
    def optimize(self,
                 train_data: Dict[str, pd.DataFrame] | pd.DataFrame,
                 val_data: Dict[str, pd.DataFrame] | pd.DataFrame,
                 strategy_class: type,
                 param_space: Dict[str, Tuple],
                 backtest_fn: Callable,
                 objective_fn: Callable,
                 constraints: List[Constraint] = None) -> OptimizationSummary:
        """Grid search optimization."""
        
        if isinstance(train_data, pd.DataFrame):
            # Single stock optimization
            return self._optimize_single(train_data, val_data, strategy_class, param_space, 
                                        backtest_fn, objective_fn, constraints)
        else:
            # Universe optimization
            return self._optimize_universe(train_data, val_data, strategy_class, param_space,
                                          backtest_fn, objective_fn, constraints)
    
    def _optimize_single(self, train_df, val_df, strategy_class, param_space, 
                        backtest_fn, objective_fn, constraints):
        """Optimize on single stock using grid search."""
        trial_id = 0
        param_ranges = self._build_param_ranges(param_space)
        total_trials = np.prod([len(v) for v in param_ranges.values()])
        
        if self.config.verbose:
            print(f"Grid Search: {total_trials} total combinations to evaluate")
        
        # Generate all parameter combinations
        import itertools
        keys = list(param_ranges.keys())
        for combo in itertools.product(*[param_ranges[k] for k in keys]):
            if self.config.n_trials and trial_id >= self.config.n_trials:
                break
            
            params = dict(zip(keys, combo))
            
            try:
                # Training evaluation
                result_train = backtest_fn(train_df, strategy_class(**params))
                
                # Check hard constraints
                if constraints:
                    satisfied, violations = self._check_constraints(result_train, constraints)
                    if not satisfied:
                        hard_violations = [v for c, v in zip(constraints, violations) if c.type == ConstraintType.HARD]
                        if hard_violations:
                            self._record_trial(trial_id, params, float('-inf'), violations=hard_violations)
                            trial_id += 1
                            continue
                
                train_score = objective_fn(result_train)
                
                # Validation evaluation
                result_val = backtest_fn(val_df, strategy_class(**params))
                val_score = objective_fn(result_val)
                
                # Combined score
                final_score = self._apply_soft_penalties(train_score, val_score, result_train, result_val, constraints)
                
                self._record_trial(trial_id, params, final_score, train_score, val_score)
                
                if self.config.verbose and (trial_id + 1) % 10 == 0:
                    print(f"  Evaluated {trial_id + 1}/{min(total_trials, self.config.n_trials)} combinations")
                    
            except Exception as e:
                self._record_trial(trial_id, params, float('-inf'), error=str(e))
            
            trial_id += 1
        
        return self._build_summary()
    
    def _optimize_universe(self, train_data, val_data, strategy_class, param_space,
                          backtest_fn, objective_fn, constraints):
        """Optimize across multiple stocks using grid search."""
        trial_id = 0
        param_ranges = self._build_param_ranges(param_space)
        symbols = list(train_data.keys())
        
        import itertools
        keys = list(param_ranges.keys())
        
        for combo in itertools.product(*[param_ranges[k] for k in keys]):
            if self.config.n_trials and trial_id >= self.config.n_trials:
                break
            
            params = dict(zip(keys, combo))
            train_scores, val_scores, per_stock_metrics = [], [], {}
            
            for symbol in symbols:
                try:
                    result_train = backtest_fn(train_data[symbol], strategy_class(**params))
                    
                    # Check constraints
                    if constraints:
                        satisfied, violations = self._check_constraints(result_train, constraints)
                        if not satisfied:
                            hard_violations = [v for c, v in zip(constraints, violations) if c.type == ConstraintType.HARD]
                            if hard_violations:
                                continue
                    
                    result_val = backtest_fn(val_data[symbol], strategy_class(**params))
                    
                    train_score = objective_fn(result_train)
                    val_score = objective_fn(result_val)
                    
                    train_scores.append(train_score)
                    val_scores.append(val_score)
                    
                    per_stock_metrics[symbol] = {
                        'train_score': train_score,
                        'val_score': val_score,
                        'train_trades': result_train.get('num_trades', 0),
                        'val_trades': result_val.get('num_trades', 0)
                    }
                except Exception:
                    continue
            
            # Require minimum stocks
            if len(train_scores) < len(symbols) / 2:
                final_score = float('-inf')
            else:
                avg_val = np.mean(val_scores)
                final_score = avg_val
                
                # Apply soft penalties
                if constraints:
                    for constraint in constraints:
                        if constraint.type == ConstraintType.SOFT and constraint.name == 'consistency':
                            consistency = np.std(val_scores)
                            final_score -= 0.1 * consistency
            
            self._record_trial(trial_id, params, final_score, 
                             np.mean(train_scores) if train_scores else np.nan,
                             np.mean(val_scores) if val_scores else np.nan,
                             metrics={'per_stock': per_stock_metrics})
            
            trial_id += 1
        
        return self._build_summary()
    
    def _build_param_ranges(self, param_space: Dict[str, Tuple]) -> Dict[str, List]:
        """Convert param_space into list of ranges."""
        ranges = {}
        for name, (ptype, *bounds) in param_space.items():
            if ptype == 'int':
                ranges[name] = list(range(int(bounds[0]), int(bounds[1]) + 1))
            elif ptype == 'float':
                # For float, create a reasonable number of steps (10 by default)
                n_steps = 10
                ranges[name] = list(np.linspace(bounds[0], bounds[1], n_steps))
            elif ptype == 'categorical':
                ranges[name] = bounds[0]  # bounds[0] should be a list of categories
        return ranges
    
    def _apply_soft_penalties(self, train_score, val_score, result_train, result_val, constraints):
        """Apply soft constraint penalties."""
        score = val_score
        
        if constraints:
            for constraint in constraints:
                if constraint.type == ConstraintType.SOFT:
                    if constraint.name == 'overfit':
                        score -= 0.3 * max(0, train_score - val_score)
        
        return score
    
    def _build_summary(self) -> OptimizationSummary:
        """Build optimization summary."""
        valid_trials = [t for t in self.trials if t.is_valid()]
        scores = [t.score for t in valid_trials]
        
        return OptimizationSummary(
            best_params=self.best_result.params if self.best_result else {},
            best_score=self.best_result.score if self.best_result else float('-inf'),
            best_trial_id=self.best_result.trial_id if self.best_result else -1,
            n_trials=len(self.trials),
            n_valid_trials=len(valid_trials),
            direction=self.config.direction,
            mean_score=np.mean(scores) if scores else np.nan,
            std_score=np.std(scores) if scores else np.nan,
            min_score=np.min(scores) if scores else np.nan,
            max_score=np.max(scores) if scores else np.nan
        )


# =============================================================================
# BAYESIAN OPTIMIZER (Using Gaussian Process - No External Dependencies)
# =============================================================================

class BayesianOptimizer(OptimizerBase):
    """Bayesian optimization using Gaussian Process (scikit-learn)."""
    
    def __init__(self, config: OptimizerConfig):
        super().__init__(config)
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern
            self.gp_class = GaussianProcessRegressor
            self.kernel_class = Matern
        except ImportError:
            raise ImportError("scikit-learn required: pip install scikit-learn")
    
    def optimize(self,
                 train_data: Dict[str, pd.DataFrame] | pd.DataFrame,
                 val_data: Dict[str, pd.DataFrame] | pd.DataFrame,
                 strategy_class: type,
                 param_space: Dict[str, Tuple],
                 backtest_fn: Callable,
                 objective_fn: Callable,
                 constraints: List[Constraint] = None) -> OptimizationSummary:
        """Bayesian optimization with Gaussian Process."""
        
        if isinstance(train_data, pd.DataFrame):
            # Single stock optimization
            return self._optimize_single(train_data, val_data, strategy_class, param_space, 
                                        backtest_fn, objective_fn, constraints)
        else:
            # Universe optimization
            return self._optimize_universe(train_data, val_data, strategy_class, param_space,
                                          backtest_fn, objective_fn, constraints)
    
    def _optimize_single(self, train_df, val_df, strategy_class, param_space, 
                        backtest_fn, objective_fn, constraints):
        """Optimize on single stock using Gaussian Process."""
        import random
        
        param_ranges = self._build_param_ranges(param_space)
        param_names = list(param_ranges.keys())
        
        # Track history for GP
        X_history = []  # Parameter values (normalized)
        y_history = []  # Scores
        
        best_score = float('-inf')
        best_params = None
        no_improve_count = 0
        
        # Initial random exploration
        n_initial = min(5, self.config.n_trials // 2)
        
        for trial_id in range(self.config.n_trials):
            if trial_id < n_initial:
                # Random sampling for initial exploration
                params = {name: random.choice(param_ranges[name]) for name in param_names}
            else:
                # GP-based sampling
                if len(y_history) >= 2:
                    # Fit GP on history
                    gp = self.gp_class(kernel=self.kernel_class(nu=2.5), random_state=self.config.seed)
                    gp.fit(X_history, y_history)
                    
                    # Suggest next point using expected improvement
                    candidates = self._generate_candidates(param_ranges, param_names, n_candidates=10)
                    candidate_X = [self._normalize_params(c, param_ranges, param_names) for c in candidates]
                    
                    # Get predictions and uncertainties
                    means, stds = gp.predict(candidate_X, return_std=True)
                    
                    # Expected improvement acquisition function
                    best_y = max(y_history)
                    ei = self._expected_improvement(means, stds, best_y)
                    
                    # Select candidate with highest EI
                    best_candidate_idx = np.argmax(ei)
                    params = candidates[best_candidate_idx]
                else:
                    # Not enough history yet
                    params = {name: random.choice(param_ranges[name]) for name in param_names}
            
            try:
                # Training evaluation
                result_train = backtest_fn(train_df, strategy_class(**params))
                
                # Check hard constraints
                if constraints:
                    satisfied, violations = self._check_constraints(result_train, constraints)
                    if not satisfied:
                        hard_violations = [v for c, v in zip(constraints, violations) if c.type == ConstraintType.HARD]
                        if hard_violations:
                            self._record_trial(trial_id, params, float('-inf'), violations=hard_violations)
                            no_improve_count += 1
                            continue
                
                train_score = objective_fn(result_train)
                
                # Validation evaluation
                result_val = backtest_fn(val_df, strategy_class(**params))
                val_score = objective_fn(result_val)
                
                # Combined score
                final_score = self._apply_soft_penalties(train_score, val_score, result_train, result_val, constraints)
                
                # Record trial
                self._record_trial(trial_id, params, final_score, train_score, val_score)
                
                # Update history
                X_history.append(self._normalize_params(params, param_ranges, param_names))
                y_history.append(final_score)
                
                # Track best
                if final_score > best_score:
                    best_score = final_score
                    best_params = params
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                if self.config.verbose and (trial_id + 1) % 10 == 0:
                    print(f"  Evaluated {trial_id + 1}/{self.config.n_trials} trials, Best: {best_score:.4f}")
                
                # Early stopping
                if self.config.early_stopping_patience and no_improve_count >= self.config.early_stopping_patience:
                    if self.config.verbose:
                        print(f"  Early stopping at trial {trial_id + 1}")
                    break
                    
            except Exception as e:
                self._record_trial(trial_id, params, float('-inf'), error=str(e))
                no_improve_count += 1
        
        return self._build_summary()
    
    def _optimize_universe(self, train_data, val_data, strategy_class, param_space,
                          backtest_fn, objective_fn, constraints):
        """Optimize across multiple stocks using Gaussian Process."""
        import random
        
        param_ranges = self._build_param_ranges(param_space)
        param_names = list(param_ranges.keys())
        symbols = list(train_data.keys())
        
        # Track history for GP
        X_history = []
        y_history = []
        
        best_score = float('-inf')
        best_params = None
        no_improve_count = 0
        
        # Initial random exploration
        n_initial = min(5, self.config.n_trials // 2)
        
        for trial_id in range(self.config.n_trials):
            if trial_id < n_initial:
                # Random sampling
                params = {name: random.choice(param_ranges[name]) for name in param_names}
            else:
                # GP-based sampling
                if len(y_history) >= 2:
                    gp = self.gp_class(kernel=self.kernel_class(nu=2.5), random_state=self.config.seed)
                    gp.fit(X_history, y_history)
                    
                    candidates = self._generate_candidates(param_ranges, param_names, n_candidates=10)
                    candidate_X = [self._normalize_params(c, param_ranges, param_names) for c in candidates]
                    
                    means, stds = gp.predict(candidate_X, return_std=True)
                    best_y = max(y_history)
                    ei = self._expected_improvement(means, stds, best_y)
                    
                    best_candidate_idx = np.argmax(ei)
                    params = candidates[best_candidate_idx]
                else:
                    params = {name: random.choice(param_ranges[name]) for name in param_names}
            
            train_scores, val_scores, per_stock_metrics = [], [], {}
            
            for symbol in symbols:
                try:
                    result_train = backtest_fn(train_data[symbol], strategy_class(**params))
                    
                    # Check constraints
                    if constraints:
                        satisfied, violations = self._check_constraints(result_train, constraints)
                        if not satisfied:
                            hard_violations = [v for c, v in zip(constraints, violations) if c.type == ConstraintType.HARD]
                            if hard_violations:
                                continue
                    
                    result_val = backtest_fn(val_data[symbol], strategy_class(**params))
                    
                    train_score = objective_fn(result_train)
                    val_score = objective_fn(result_val)
                    
                    train_scores.append(train_score)
                    val_scores.append(val_score)
                    
                    per_stock_metrics[symbol] = {
                        'train_score': train_score,
                        'val_score': val_score,
                        'train_trades': result_train.get('num_trades', 0),
                        'val_trades': result_val.get('num_trades', 0)
                    }
                except Exception:
                    continue
            
            # Require minimum stocks
            if len(train_scores) < len(symbols) / 2:
                final_score = float('-inf')
            else:
                avg_val = np.mean(val_scores)
                final_score = avg_val
                
                # Apply soft penalties
                if constraints:
                    for constraint in constraints:
                        if constraint.type == ConstraintType.SOFT and constraint.name == 'consistency':
                            consistency = np.std(val_scores)
                            final_score -= 0.1 * consistency
            
            # Record trial
            self._record_trial(trial_id, params, final_score, 
                             np.mean(train_scores) if train_scores else np.nan,
                             np.mean(val_scores) if val_scores else np.nan,
                             metrics={'per_stock': per_stock_metrics})
            
            # Update history
            if len(train_scores) > 0:
                X_history.append(self._normalize_params(params, param_ranges, param_names))
                y_history.append(final_score)
                
                if final_score > best_score:
                    best_score = final_score
                    best_params = params
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            
            # Early stopping
            if self.config.early_stopping_patience and no_improve_count >= self.config.early_stopping_patience:
                if self.config.verbose:
                    print(f"  Early stopping at trial {trial_id + 1}")
                break
        
        return self._build_summary()
    
    def _build_param_ranges(self, param_space: Dict[str, Tuple]) -> Dict[str, List]:
        """Convert param_space into list of ranges."""
        ranges = {}
        for name, (ptype, *bounds) in param_space.items():
            if ptype == 'int':
                ranges[name] = list(range(int(bounds[0]), int(bounds[1]) + 1))
            elif ptype == 'float':
                n_steps = 10
                ranges[name] = list(np.linspace(bounds[0], bounds[1], n_steps))
            elif ptype == 'categorical':
                ranges[name] = bounds[0]
        return ranges
    
    def _normalize_params(self, params: Dict, param_ranges: Dict, param_names: List) -> np.ndarray:
        """Normalize parameters to [0, 1] range."""
        normalized = []
        for name in param_names:
            value = params[name]
            param_list = param_ranges[name]
            
            # Find min and max
            if isinstance(param_list[0], (int, float)):
                p_min, p_max = min(param_list), max(param_list)
                norm_val = (value - p_min) / (p_max - p_min + 1e-8)
            else:
                # Categorical
                norm_val = param_list.index(value) / len(param_list)
            
            normalized.append(norm_val)
        
        return np.array(normalized).reshape(1, -1)[0]
    
    def _denormalize_params(self, normalized: np.ndarray, param_ranges: Dict, param_names: List) -> Dict:
        """Denormalize parameters from [0, 1] back to original range."""
        params = {}
        for i, name in enumerate(param_names):
            norm_val = normalized[i]
            param_list = param_ranges[name]
            
            if isinstance(param_list[0], (int, float)):
                p_min, p_max = min(param_list), max(param_list)
                value = p_min + norm_val * (p_max - p_min)
                # Round to nearest valid value
                params[name] = param_list[np.argmin(np.abs(np.array(param_list) - value))]
            else:
                # Categorical
                idx = int(norm_val * len(param_list))
                idx = min(idx, len(param_list) - 1)
                params[name] = param_list[idx]
        
        return params
    
    def _generate_candidates(self, param_ranges: Dict, param_names: List, n_candidates: int = 10) -> List[Dict]:
        """Generate candidate parameter combinations."""
        import random
        candidates = []
        for _ in range(n_candidates):
            params = {name: random.choice(param_ranges[name]) for name in param_names}
            candidates.append(params)
        return candidates
    
    def _expected_improvement(self, means: np.ndarray, stds: np.ndarray, best_y: float, xi: float = 0.0) -> np.ndarray:
        """Calculate Expected Improvement acquisition function."""
        from scipy.stats import norm
        
        means = np.atleast_1d(means)
        stds = np.atleast_1d(stds)
        
        # Avoid division by zero
        stds = np.maximum(stds, 1e-9)
        
        # Calculate improvement
        improvement = means - best_y - xi
        Z = improvement / stds
        ei = improvement * norm.cdf(Z) + stds * norm.pdf(Z)
        ei[stds == 0.0] = 0.0
        
        return ei
    
    def _apply_soft_penalties(self, train_score, val_score, result_train, result_val, constraints):
        """Apply soft constraint penalties."""
        score = val_score
        
        if constraints:
            for constraint in constraints:
                if constraint.type == ConstraintType.SOFT:
                    if constraint.name == 'overfit':
                        score -= 0.3 * max(0, train_score - val_score)
        
        return score
    
    def _build_summary(self) -> OptimizationSummary:
        """Build optimization summary."""
        valid_trials = [t for t in self.trials if t.is_valid()]
        scores = [t.score for t in valid_trials]
        
        return OptimizationSummary(
            best_params=self.best_result.params if self.best_result else {},
            best_score=self.best_result.score if self.best_result else float('-inf'),
            best_trial_id=self.best_result.trial_id if self.best_result else -1,
            n_trials=len(self.trials),
            n_valid_trials=len(valid_trials),
            direction=self.config.direction,
            mean_score=np.mean(scores) if scores else np.nan,
            std_score=np.std(scores) if scores else np.nan,
            min_score=np.min(scores) if scores else np.nan,
            max_score=np.max(scores) if scores else np.nan
        )


# =============================================================================
# OPTUNA OPTIMIZER (Advanced Bayesian with Optuna TPE)
# =============================================================================

class OptunaOptimizer(OptimizerBase):
    """Advanced Bayesian optimization using Optuna TPE sampler."""
    
    def __init__(self, config: OptimizerConfig):
        super().__init__(config)
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            self.optuna = optuna
        except ImportError:
            raise ImportError("Optuna required: pip install optuna")
    
    def optimize(self,
                 train_data: Dict[str, pd.DataFrame] | pd.DataFrame,
                 val_data: Dict[str, pd.DataFrame] | pd.DataFrame,
                 strategy_class: type,
                 param_space: Dict[str, Tuple],
                 backtest_fn: Callable,
                 objective_fn: Callable,
                 constraints: List[Constraint] = None) -> OptimizationSummary:
        """Bayesian optimization with Optuna TPE sampler."""
        
        if isinstance(train_data, pd.DataFrame):
            # Single stock optimization
            return self._optimize_single(train_data, val_data, strategy_class, param_space, 
                                        backtest_fn, objective_fn, constraints)
        else:
            # Universe optimization
            return self._optimize_universe(train_data, val_data, strategy_class, param_space,
                                          backtest_fn, objective_fn, constraints)
    
    def _optimize_single(self, train_df, val_df, strategy_class, param_space, 
                        backtest_fn, objective_fn, constraints):
        """Optimize on single stock."""
        trial_id = [0]
        
        def objective(trial):
            params = self._sample_params(trial, param_space)
            
            try:
                result_train = backtest_fn(train_df, strategy_class(**params))
                
                # Check hard constraints
                if constraints:
                    satisfied, violations = self._check_constraints(result_train, constraints)
                    if not satisfied:
                        hard_violations = [v for c, v in zip(constraints, violations) if c.type == ConstraintType.HARD]
                        if hard_violations:
                            self._record_trial(trial_id[0], params, float('-inf'), violations=hard_violations)
                            trial_id[0] += 1
                            return float('-inf')
                
                train_score = objective_fn(result_train)
                
                # Validation
                result_val = backtest_fn(val_df, strategy_class(**params))
                val_score = objective_fn(result_val)
                
                # Combined score
                final_score = self._apply_soft_penalties(train_score, val_score, result_train, result_val, constraints)
                
                # Pruning
                if self.config.use_pruning:
                    trial.report(final_score, step=0)
                    if trial.should_prune():
                        raise self.optuna.TrialPruned()
                
                self._record_trial(trial_id[0], params, final_score, train_score, val_score)
                trial_id[0] += 1
                return final_score
                
            except self.optuna.TrialPruned:
                raise
            except Exception as e:
                self._record_trial(trial_id[0], params, float('-inf'), error=str(e))
                trial_id[0] += 1
                return float('-inf')
        
        # Run optimization
        pruner = self.optuna.pruners.MedianPruner(n_startup_trials=5) if self.config.use_pruning else self.optuna.pruners.NopPruner()
        study = self.optuna.create_study(
            direction=self.config.direction.value,
            pruner=pruner,
            sampler=self.optuna.samplers.TPESampler(seed=self.config.seed)
        )
        
        callbacks = []
        if self.config.early_stopping_patience:
            callbacks.append(self._create_early_stopper())
        
        study.optimize(objective, n_trials=self.config.n_trials, n_jobs=self.config.n_jobs,
                      timeout=self.config.timeout, callbacks=callbacks, show_progress_bar=self.config.verbose)
        
        # Build summary
        return self._build_summary(study)
    
    def _optimize_universe(self, train_data, val_data, strategy_class, param_space,
                          backtest_fn, objective_fn, constraints):
        """Optimize across multiple stocks."""
        symbols = list(train_data.keys())
        trial_id = [0]
        
        def objective(trial):
            params = self._sample_params(trial, param_space)
            train_scores, val_scores, per_stock_metrics = [], [], {}
            
            for symbol in symbols:
                try:
                    result_train = backtest_fn(train_data[symbol], strategy_class(**params))
                    
                    # Check constraints
                    if constraints:
                        satisfied, violations = self._check_constraints(result_train, constraints)
                        if not satisfied:
                            hard_violations = [v for c, v in zip(constraints, violations) if c.type == ConstraintType.HARD]
                            if hard_violations:
                                continue
                    
                    result_val = backtest_fn(val_data[symbol], strategy_class(**params))
                    
                    train_score = objective_fn(result_train)
                    val_score = objective_fn(result_val)
                    
                    train_scores.append(train_score)
                    val_scores.append(val_score)
                    
                    per_stock_metrics[symbol] = {
                        'train_score': train_score,
                        'val_score': val_score,
                        'train_trades': result_train.get('num_trades', 0),
                        'val_trades': result_val.get('num_trades', 0)
                    }
                    
                except Exception:
                    continue
            
            # Require minimum stocks
            if len(train_scores) < len(symbols) / 2:
                return float('-inf')
            
            avg_val = np.mean(val_scores)
            final_score = avg_val
            
            # Apply soft penalties
            if constraints:
                for constraint in constraints:
                    if constraint.type == ConstraintType.SOFT and constraint.name == 'consistency':
                        consistency = np.std(val_scores)
                        final_score -= 0.1 * consistency
            
            self._record_trial(trial_id[0], params, final_score, np.mean(train_scores), avg_val,
                             metrics={'per_stock': per_stock_metrics})
            trial_id[0] += 1
            return final_score
        
        # Run optimization
        pruner = self.optuna.pruners.MedianPruner(n_startup_trials=5) if self.config.use_pruning else self.optuna.pruners.NopPruner()
        study = self.optuna.create_study(
            direction=self.config.direction.value,
            pruner=pruner,
            sampler=self.optuna.samplers.TPESampler(seed=self.config.seed)
        )
        
        callbacks = []
        if self.config.early_stopping_patience:
            callbacks.append(self._create_early_stopper())
        
        study.optimize(objective, n_trials=self.config.n_trials, n_jobs=self.config.n_jobs,
                      timeout=self.config.timeout, callbacks=callbacks, show_progress_bar=self.config.verbose)
        
        return self._build_summary(study)
    
    def _sample_params(self, trial, param_space: Dict[str, Tuple]) -> Dict:
        """Sample parameters from space."""
        params = {}
        for name, (ptype, *bounds) in param_space.items():
            if ptype == 'int':
                params[name] = trial.suggest_int(name, bounds[0], bounds[1])
            elif ptype == 'float':
                params[name] = trial.suggest_float(name, bounds[0], bounds[1])
            elif ptype == 'categorical':
                params[name] = trial.suggest_categorical(name, bounds[0])
        return params
    
    def _create_early_stopper(self):
        """Create early stopping callback."""
        patience = self.config.early_stopping_patience
        
        class EarlyStopper:
            def __init__(self):
                self.best_value = float('-inf')
                self.patience_counter = 0
            
            def __call__(self, study, trial):
                if study.best_value > self.best_value + self.config.early_stopping_min_delta:
                    self.best_value = study.best_value
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= patience:
                        study.stop()
        
        return EarlyStopper()
    
    def _apply_soft_penalties(self, train_score, val_score, result_train, result_val, constraints):
        """Apply soft constraint penalties."""
        score = val_score
        
        if constraints:
            for constraint in constraints:
                if constraint.type == ConstraintType.SOFT:
                    if constraint.name == 'overfit':
                        score -= 0.3 * max(0, train_score - val_score)
        
        return score
    
    def _build_summary(self, study) -> OptimizationSummary:
        """Build optimization summary from study."""
        valid_trials = [t for t in self.trials if t.is_valid()]
        scores = [t.score for t in valid_trials]
        
        return OptimizationSummary(
            best_params=self.best_result.params,
            best_score=self.best_result.score,
            best_trial_id=self.best_result.trial_id,
            n_trials=len(self.trials),
            n_valid_trials=len(valid_trials),
            direction=self.config.direction,
            mean_score=np.mean(scores) if scores else np.nan,
            std_score=np.std(scores) if scores else np.nan,
            min_score=np.min(scores) if scores else np.nan,
            max_score=np.max(scores) if scores else np.nan
        )


__all__ = [
    'OptimizationStrategy', 'ObjectiveDirection', 'ConstraintType',
    'OptimizationResult', 'OptimizationSummary', 'OptimizerConfig',
    'Constraint', 'TradeCountConstraint', 'SharpeConstraint', 'DrawdownConstraint', 'ReturnConstraint',
    'ObjectiveBuilder',
    'OptimizerBase', 'GridSearchOptimizer', 'BayesianOptimizer', 'OptunaOptimizer'
]
