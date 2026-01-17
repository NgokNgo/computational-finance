"""
Optimizer Module - Parameter Optimization Framework

Provides multiple optimization strategies:
- GridSearchOptimizer: Exhaustive parameter grid search
- BayesianOptimizer: Gaussian Process-based Bayesian optimization
- OptunaOptimizer: Advanced TPE sampling with Optuna
- RandomSearchOptimizer: Random parameter sampling
- GeneticAlgorithmOptimizer: Evolutionary algorithm optimization

Quick Start:
    from utils.optimizer import BayesianOptimizer, OptimizerConfig
    
    config = OptimizerConfig(n_trials=100)
    optimizer = BayesianOptimizer(config)
    summary = optimizer.optimize(...)
"""

# Core classes
from .optimizer_core import (
    OptimizationStrategy,
    ObjectiveDirection,
    ConstraintType,
    OptimizationResult,
    OptimizationSummary,
    OptimizerConfig,
    Constraint,
    TradeCountConstraint,
    SharpeConstraint,
    DrawdownConstraint,
    ReturnConstraint,
    ObjectiveBuilder,
    OptimizerBase,
    GridSearchOptimizer,
    BayesianOptimizer,
    OptunaOptimizer,
)

# Alternative strategies
from .optimizer_strategies import (
    RandomSearchOptimizer,
    GeneticAlgorithmOptimizer,
    OptimizerFactory,
)

# Analysis tools
from .optimizer_analyzer import (
    ResultsAnalyzer,
    ParameterSensitivity,
    RobustnessTester,
)

# High-level API
from .optimizer_api import (
    Optimizer,
    quick_optimize,
)

__all__ = [
    # Enums
    'OptimizationStrategy',
    'ObjectiveDirection',
    'ConstraintType',
    # Models
    'OptimizationResult',
    'OptimizationSummary',
    'OptimizerConfig',
    # Constraints
    'Constraint',
    'TradeCountConstraint',
    'SharpeConstraint',
    'DrawdownConstraint',
    'ReturnConstraint',
    # Objectives
    'ObjectiveBuilder',
    # Base & Core Optimizers
    'OptimizerBase',
    'GridSearchOptimizer',
    'BayesianOptimizer',
    'OptunaOptimizer',
    # Alternative Strategies
    'RandomSearchOptimizer',
    'GeneticAlgorithmOptimizer',
    'OptimizerFactory',
    # Analysis
    'ResultsAnalyzer',
    'ParameterSensitivity',
    'RobustnessTester',
    # High-level API
    'Optimizer',
    'quick_optimize',
]

__version__ = '2.0.0'
__doc__ = """
Optimizer Module v2.0.0

Changes in v2.0.0:
- BayesianOptimizer now uses Gaussian Process (no Optuna required)
- New OptunaOptimizer for advanced TPE sampling (optional Optuna)
- Improved architecture with modular design
- Backward compatible with v1.0.0 API
"""
