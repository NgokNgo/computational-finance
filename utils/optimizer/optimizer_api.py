"""
Unified Optimizer API - High-level interface for optimization

Lean-inspired modular optimizer with simplified API:
- Easy parameter specification
- Multiple optimization strategies
- Automatic constraint management
- Comprehensive results analysis
"""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple, Union
import time

from .optimizer_core import (
    OptimizationStrategy, ObjectiveDirection, ConstraintType,
    OptimizerConfig, Constraint, ObjectiveBuilder,
    BayesianOptimizer
)
from .optimizer_strategies import OptimizerFactory
from .optimizer_analyzer import ResultsAnalyzer, RobustnessTester

import numpy as np
import pandas as pd


# =============================================================================
# HIGH-LEVEL OPTIMIZER API
# =============================================================================

class Optimizer:
    """
    High-level optimizer API for strategy parameter optimization.
    
    Inspired by QuantConnect/Lean architecture with:
    - Multiple optimization strategies
    - Constraint management
    - Automatic parameter validation
    - Comprehensive analysis
    
    Example:
        >>> opt = Optimizer(strategy=OptimizationStrategy.BAYESIAN)
        >>> opt.add_constraint('trade_count', min_trades=10, max_trades=100)
        >>> opt.set_objective('sharpe', overfit_penalty=0.3)
        >>> result = opt.optimize(
        ...     train_data, val_data, strategy_class,
        ...     param_space, backtest_fn, n_trials=100
        ... )
    """
    
    def __init__(self,
                 strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN,
                 direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE,
                 n_trials: int = 100,
                 n_jobs: int = 1,
                 timeout: Optional[int] = None,
                 seed: int = 42,
                 verbose: bool = True):
        """
        Initialize optimizer.
        
        Args:
            strategy: Optimization strategy
            direction: Maximize or minimize
            n_trials: Number of trials
            n_jobs: Number of parallel jobs
            timeout: Timeout in seconds
            seed: Random seed
            verbose: Print progress
        """
        self.config = OptimizerConfig(
            strategy=strategy,
            direction=direction,
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout,
            seed=seed,
            verbose=verbose
        )
        self.constraints: List[Constraint] = []
        self.objective_fn: Optional[Callable] = None
        self._optimizer_instance: Optional[OptimizerFactory] = None
    
    def add_constraint(self, constraint_type: str, **kwargs) -> Optimizer:
        """
        Add constraint to optimization.
        
        Args:
            constraint_type: 'trade_count', 'sharpe', 'drawdown', 'return'
            **kwargs: Constraint parameters
        
        Example:
            >>> opt.add_constraint('trade_count', min_trades=10, max_trades=200)
            >>> opt.add_constraint('sharpe', min_sharpe=0.5, ctype='soft')
        """
        from .optimizer_core import (
            TradeCountConstraint, SharpeConstraint,
            DrawdownConstraint, ReturnConstraint
        )
        
        ctype = ConstraintType.HARD
        if 'ctype' in kwargs:
            ctype_str = kwargs.pop('ctype')
            ctype = ConstraintType.HARD if ctype_str == 'hard' else ConstraintType.SOFT
        
        if constraint_type == 'trade_count':
            constraint = TradeCountConstraint(
                min_trades=kwargs.get('min_trades', 10),
                max_trades=kwargs.get('max_trades'),
                ctype=ctype
            )
        elif constraint_type == 'sharpe':
            constraint = SharpeConstraint(
                min_sharpe=kwargs.get('min_sharpe', 0.5),
                ctype=ctype
            )
        elif constraint_type == 'drawdown':
            constraint = DrawdownConstraint(
                max_drawdown=kwargs.get('max_drawdown', -0.2),
                ctype=ctype
            )
        elif constraint_type == 'return':
            constraint = ReturnConstraint(
                min_return=kwargs.get('min_return', 0.0),
                ctype=ctype
            )
        else:
            raise ValueError(f"Unknown constraint type: {constraint_type}")
        
        self.constraints.append(constraint)
        return self
    
    def set_objective(self, metric: str,
                     sharpe_weight: float = 0.5,
                     return_weight: float = 0.3,
                     drawdown_weight: float = 0.2,
                     overfit_penalty: float = 0.3,
                     consistency_penalty: float = 0.1,
                     trade_variance_penalty: float = 0.0) -> Optimizer:
        """
        Set objective function.
        
        Args:
            metric: 'sharpe', 'return', 'drawdown', 'combined'
            sharpe_weight: Weight for Sharpe ratio
            return_weight: Weight for returns
            drawdown_weight: Weight for max drawdown
            overfit_penalty: Penalty for overfitting
            consistency_penalty: Penalty for inconsistent results
            trade_variance_penalty: Penalty for trade variance
        
        Example:
            >>> opt.set_objective('combined', sharpe_weight=0.4,
            ...                  overfit_penalty=0.2)
        """
        builder = ObjectiveBuilder(self.config.direction)
        
        if metric == 'sharpe':
            builder.add_component(lambda r: r.get('sharpe_ratio', 0), weight=1.0)
        elif metric == 'return':
            builder.add_component(lambda r: r.get('total_return', 0), weight=1.0)
        elif metric == 'combined':
            builder.add_component(lambda r: r.get('sharpe_ratio', 0), weight=sharpe_weight)
            builder.add_component(lambda r: r.get('total_return', 0), weight=return_weight)
            # Negative for drawdown penalty
            builder.add_component(lambda r: -abs(r.get('max_drawdown', 0)), weight=drawdown_weight)
        else:
            # Default to sharpe
            builder.add_component(lambda r: r.get('sharpe_ratio', 0), weight=1.0)
        
        # Add penalties
        if overfit_penalty > 0:
            builder.add_penalty('overfit', overfit_penalty)
        if consistency_penalty > 0:
            builder.add_penalty('consistency', consistency_penalty)
        if trade_variance_penalty > 0:
            builder.add_penalty('trade_variance', trade_variance_penalty)
        
        self.objective_fn = builder.build()
        return self
    
    def optimize(self,
                 train_data: Union[Dict[str, pd.DataFrame], pd.DataFrame],
                 val_data: Union[Dict[str, pd.DataFrame], pd.DataFrame],
                 strategy_class: type,
                 param_space: Dict[str, Tuple],
                 backtest_fn: Callable,
                 return_analyzer: bool = True) -> Union[Dict, Tuple]:
        """
        Run optimization.
        
        Args:
            train_data: Training data (single stock or universe dict)
            val_data: Validation data
            strategy_class: Strategy class to optimize
            param_space: Parameter space specification
            backtest_fn: Backtest function
            return_analyzer: Return results with analyzer
        
        Returns:
            Dictionary with results, or tuple (results, analyzer) if return_analyzer=True
        
        Example:
            >>> param_space = {
            ...     'period': ('int', 7, 21),
            ...     'oversold': ('int', 20, 35),
            ...     'overbought': ('int', 65, 80)
            ... }
            >>> result = opt.optimize(train_data, val_data, 
            ...                        RSIMomentum, param_space, backtest)
        """
        
        # Set default objective if not set
        if self.objective_fn is None:
            self.set_objective('sharpe')
        
        # Create optimizer
        optimizer = OptimizerFactory.create(self.config)
        
        if self.config.verbose:
            print("\n" + "=" * 80)
            print(f"OPTIMIZATION START - Strategy: {self.config.strategy.value}")
            print("=" * 80)
            print(f"Constraints: {len(self.constraints)}")
            print(f"Trials: {self.config.n_trials}")
            print(f"Direction: {self.config.direction.value}")
        
        start_time = time.time()
        
        # Run optimization
        summary = optimizer.optimize(
            train_data=train_data,
            val_data=val_data,
            strategy_class=strategy_class,
            param_space=param_space,
            backtest_fn=backtest_fn,
            objective_fn=self.objective_fn,
            constraints=self.constraints
        )
        
        elapsed = time.time() - start_time
        
        result = {
            'best_params': summary.best_params,
            'best_score': summary.best_score,
            'summary': summary.to_dict(),
            'trials': len(optimizer.trials),
            'elapsed_time': elapsed
        }
        
        if self.config.verbose:
            print(f"\nOptimization completed in {elapsed:.1f}s")
            print(f"Best Score: {summary.best_score:.6f}")
            print(f"Best Params: {summary.best_params}")
            print("=" * 80)
        
        if return_analyzer:
            analyzer = ResultsAnalyzer(optimizer.trials, summary)
            return result, analyzer
        else:
            return result
    
    def test_robustness(self,
                       test_data: Union[Dict[str, pd.DataFrame], pd.DataFrame],
                       best_params: Dict,
                       strategy_class: type,
                       backtest_fn: Callable,
                       test_type: str = 'bootstrap',
                       **kwargs) -> Dict:
        """
        Test robustness of optimized parameters.
        
        Args:
            test_data: Test data for robustness testing
            best_params: Best parameters from optimization
            strategy_class: Strategy class
            backtest_fn: Backtest function
            test_type: 'bootstrap', 'walk_forward', 'monte_carlo'
            **kwargs: Additional arguments for test
        
        Returns:
            Robustness test results
        
        Example:
            >>> robustness = opt.test_robustness(
            ...     test_data, best_params, RSIMomentum,
            ...     backtest, test_type='bootstrap', n_bootstrap=100
            ... )
        """
        
        if isinstance(test_data, dict):
            # Use first symbol for robustness test
            test_df = list(test_data.values())[0]
        else:
            test_df = test_data
        
        if test_type == 'bootstrap':
            return RobustnessTester.bootstrap_test(
                test_df, strategy_class, best_params, backtest_fn,
                n_bootstrap=kwargs.get('n_bootstrap', 100),
                sample_ratio=kwargs.get('sample_ratio', 0.8)
            )
        elif test_type == 'walk_forward':
            return RobustnessTester.walk_forward_test(
                test_df, strategy_class, best_params, backtest_fn,
                n_periods=kwargs.get('n_periods', 5)
            )
        elif test_type == 'monte_carlo':
            return RobustnessTester.monte_carlo_test(
                test_df, strategy_class, best_params, backtest_fn,
                n_simulations=kwargs.get('n_simulations', 100)
            )
        else:
            raise ValueError(f"Unknown test type: {test_type}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_optimize(train_data: Union[Dict[str, pd.DataFrame], pd.DataFrame],
                  val_data: Union[Dict[str, pd.DataFrame], pd.DataFrame],
                  strategy_class: type,
                  param_space: Dict[str, Tuple],
                  backtest_fn: Callable,
                  n_trials: int = 100,
                  objective: str = 'sharpe',
                  strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN,
                  **kwargs) -> Dict:
    """
    Quick optimization with sensible defaults.
    
    Example:
        >>> result = quick_optimize(
        ...     train_data, val_data, RSIMomentum,
        ...     param_space, backtest_fn,
        ...     n_trials=100, objective='combined'
        ... )
    """
    opt = Optimizer(
        strategy=strategy,
        n_trials=n_trials,
        verbose=True
    )
    
    opt.add_constraint('trade_count', min_trades=10, ctype='hard')
    opt.set_objective(objective, **kwargs)
    
    result, analyzer = opt.optimize(
        train_data, val_data, strategy_class,
        param_space, backtest_fn
    )
    
    print("\n" + analyzer.generate_report())
    
    return result


__all__ = [
    'Optimizer',
    'quick_optimize',
    'OptimizationStrategy',
    'ObjectiveDirection',
    'ConstraintType'
]
