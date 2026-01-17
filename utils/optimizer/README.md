"""
Lean-inspired Modular Optimizer - README

A production-grade parameter optimization framework inspired by QuantConnect/Lean.
"""

# ============================================================================
# LEAN-INSPIRED OPTIMIZER ARCHITECTURE
# ============================================================================

## Overview

The new optimizer provides a modular, enterprise-grade framework for strategy
parameter optimization with:

- **Multiple Strategies**: Grid Search, Bayesian (Gaussian Process), Optuna TPE, Random Search, Genetic Algorithm
- **Constraint System**: Hard and soft constraints with flexible validation
- **Advanced Analysis**: Parameter sensitivity, interaction analysis, robustness testing
- **Universe Optimization**: Optimize across multiple assets simultaneously
- **Clean API**: Intuitive, chainable interface inspired by Lean framework

## Core Components

### 1. optimizer_core.py
Base classes and core functionality:
- OptimizationStrategy enum
- ObjectiveDirection enum  
- ConstraintType enum
- OptimizationResult dataclass
- OptimizationSummary dataclass
- OptimizerConfig dataclass
- Constraint base class and implementations
- ObjectiveBuilder for custom objectives
- OptimizerBase abstract base class
- GridSearchOptimizer: Exhaustive parameter grid search
- BayesianOptimizer: Gaussian Process-based Bayesian optimization (no Optuna required)
- OptunaOptimizer: Advanced Bayesian with Optuna TPE sampler

### 2. optimizer_strategies.py
Alternative optimization approaches:
- GridSearchOptimizer: Exhaustive parameter grid evaluation
- RandomSearchOptimizer: Random parameter sampling
- GeneticAlgorithmOptimizer: Evolutionary algorithm
- OptimizerFactory: Factory pattern for optimizer creation

### 3. optimizer_analyzer.py
Results analysis and robustness testing:
- ResultsAnalyzer: Comprehensive result statistics
- ParameterSensitivity: Parameter sensitivity analysis
- RobustnessTester: Bootstrap, walk-forward, Monte Carlo testing

### 4. optimizer_api.py
High-level user-facing API:
- Optimizer: Main class with fluent interface
- quick_optimize: Convenience function with defaults

## Quick Start

```python
from utils import Optimizer, OptimizationStrategy
from strategies.momentum import RSIMomentum
from strategies.base import backtest_strategy

# Create optimizer
opt = Optimizer(
    strategy=OptimizationStrategy.BAYESIAN,
    n_trials=100,
    verbose=True
)

# Add constraints
opt.add_constraint('trade_count', min_trades=10, max_trades=200, ctype='hard')
opt.add_constraint('sharpe', min_sharpe=0.5, ctype='soft')

# Set objective
opt.set_objective('sharpe', overfit_penalty=0.3)

# Parameter space
param_space = {
    'period': ('int', 7, 21),
    'oversold': ('int', 20, 35),
    'overbought': ('int', 65, 80)
}

# Run optimization
result, analyzer = opt.optimize(
    train_data, val_data, 
    RSIMomentum, param_space, 
    backtest_strategy
)

# Print analysis
print(analyzer.generate_report())

# Test robustness
robustness = opt.test_robustness(
    test_data, result['best_params'],
    RSIMomentum, backtest_strategy,
    test_type='bootstrap', n_bootstrap=100
)
```

## Features

### 1. Multiple Optimization Strategies

#### Bayesian Optimization (Gaussian Process)
- Uses scikit-learn's Gaussian Process Regressor
- Expected Improvement (EI) acquisition function
- No Optuna dependency required
- Good for moderate-sized parameter spaces
- Recommended default Bayesian optimizer

```python
opt = Optimizer(strategy=OptimizationStrategy.BAYESIAN)
```

#### Optuna Bayesian Optimization (TPE)
- Uses Optuna's Tree-structured Parzen Estimator
- Advanced exploration/exploitation with pruning
- Best for large parameter spaces
- Requires Optuna installation
- Production-grade optimization

```python
from utils.optimizer_core import OptunaOptimizer
opt = OptunaOptimizer(config)
```

#### Grid Search
- Evaluates all parameter combinations
- Exhaustive and deterministic
- Good for small parameter spaces

```python
opt = Optimizer(strategy=OptimizationStrategy.GRID_SEARCH)
```

#### Random Search
- Random parameter sampling
- Parallel evaluation friendly
- Baseline comparison method

```python
opt = Optimizer(strategy=OptimizationStrategy.RANDOM)
```

#### Genetic Algorithm
- Evolutionary approach
- Good for complex, non-convex landscapes
- Tunable mutation and crossover rates

```python
opt = Optimizer(strategy=OptimizationStrategy.GENETIC)
```

### 2. Flexible Constraint System

#### Hard Constraints
Trials rejected if violated:
```python
opt.add_constraint('trade_count', min_trades=10, max_trades=200, ctype='hard')
opt.add_constraint('drawdown', max_drawdown=-0.3, ctype='hard')
```

#### Soft Constraints
Penalty applied if violated:
```python
opt.add_constraint('sharpe', min_sharpe=0.5, ctype='soft')
opt.add_constraint('return', min_return=0.1, ctype='soft')
```

#### Supported Constraints
- `trade_count`: Number of trades
- `sharpe`: Sharpe ratio threshold
- `drawdown`: Maximum drawdown limit
- `return`: Minimum return requirement

### 3. Objective Functions

#### Predefined Metrics
```python
opt.set_objective('sharpe')        # Maximize Sharpe ratio
opt.set_objective('return')        # Maximize returns
opt.set_objective('combined', 
                  sharpe_weight=0.4,      # 40% Sharpe
                  return_weight=0.3,      # 30% Return
                  drawdown_weight=0.2)    # 20% Drawdown
```

#### Custom Objectives
```python
from utils import ObjectiveBuilder, ObjectiveDirection

builder = ObjectiveBuilder(ObjectiveDirection.MAXIMIZE)
builder.add_component(lambda r: r['sharpe_ratio'], weight=0.5)
builder.add_component(lambda r: r['calmar_ratio'], weight=0.3)
builder.add_penalty('overfit', 0.2)
builder.add_penalty('consistency', 0.1)

opt.objective_fn = builder.build()
```

### 4. Parameter Sensitivity Analysis

```python
result, analyzer = opt.optimize(...)

# Sensitivity to single parameter
sensitivity = analyzer.analyze_parameter_sensitivity('period')
print(f"Optimal: {sensitivity.optimal_value}")
print(f"Correlation: {sensitivity.correlation:.3f}")

# Parameter interactions
interaction = analyzer.analyze_parameter_interactions('period', 'oversold')
print(interaction)  # Pivot table
```

### 5. Robustness Testing

#### Bootstrap Test
```python
bootstrap_results = opt.test_robustness(
    test_data, best_params, RSIMomentum, backtest_strategy,
    test_type='bootstrap', n_bootstrap=100
)
# Check 95% confidence interval
print(f"95% CI: [{bootstrap_results['ci_lower']:.3f}, {bootstrap_results['ci_upper']:.3f}]")
```

#### Walk-Forward Test
```python
wf_results = opt.test_robustness(
    test_data, best_params, RSIMomentum, backtest_strategy,
    test_type='walk_forward', n_periods=5
)
# Check consistency across periods
print(f"Consistency: {wf_results['consistency']:.3f}")
```

#### Monte Carlo Test
```python
mc_results = opt.test_robustness(
    test_data, best_params, RSIMomentum, backtest_strategy,
    test_type='monte_carlo', n_simulations=100
)
# Check if strategy works with random order
print(f"Success Rate: {100 * mc_results['success_rate']:.1f}%")
```

### 6. Universe Optimization

Optimize parameters across multiple stocks:

```python
# Load multiple stocks
train_data = {
    'VNM': df_vnm_train,
    'VCB': df_vcb_train,
    'BID': df_bid_train
}
val_data = {
    'VNM': df_vnm_val,
    'VCB': df_vcb_val,
    'BID': df_bid_val
}

# Optimize across all stocks
result, analyzer = opt.optimize(
    train_data, val_data, RSIMomentum,
    param_space, backtest_strategy
)

# Results include per-stock metrics
per_stock = analyzer.summary.per_stock_results
```

### 7. Comprehensive Analysis

```python
result, analyzer = opt.optimize(...)

# Statistics
stats = analyzer.get_statistics()
print(f"Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")

# Top parameters
top_5 = analyzer.get_best_parameters(n=5)
for params, score in top_5:
    print(f"Score: {score:.6f}, Params: {params}")

# Overfitting analysis
overfit = analyzer.get_overfitting_analysis()
print(f"Mean Gap: {overfit['mean_gap']:.6f}")

# Constraint violations
constraints = analyzer.get_constraint_analysis()
print(f"Valid Ratio: {constraints['valid_ratio']:.1%}")

# Convergence
convergence = analyzer.get_convergence_analysis()
print(f"Improvements: {convergence['n_improvements']}")

# Full report
print(analyzer.generate_report())
```

## Lean Framework Integration

The optimizer architecture follows QuantConnect/Lean design principles:

1. **Modularity**: Each component (core, strategies, analysis) is independent
2. **Pluggability**: Easily add new optimization strategies or constraints
3. **Event-driven**: Results tracked through OptimizationResult events
4. **Comprehensive**: Covers full optimization lifecycle
5. **Production-ready**: Robust error handling and validation

## Advanced Usage

### Batch Optimization
```python
strategies = [
    OptimizationStrategy.BAYESIAN,
    OptimizationStrategy.GRID_SEARCH,
    OptimizationStrategy.RANDOM
]

results = {}
for strategy in strategies:
    opt = Optimizer(strategy=strategy, n_trials=100)
    opt.set_objective('sharpe')
    result, analyzer = opt.optimize(...)
    results[strategy.value] = result
```

### Early Stopping
```python
opt = Optimizer(n_trials=100)
opt.config.early_stopping_patience = 20
opt.config.early_stopping_min_delta = 1e-4
```

### Parallel Evaluation
```python
opt = Optimizer(n_jobs=4)  # Use 4 cores
```

### Custom Constraint
```python
from utils import Constraint, ConstraintType

class CustomConstraint(Constraint):
    def __init__(self):
        super().__init__("custom", ConstraintType.SOFT)
    
    def check(self, result):
        # Your logic here
        if not some_condition(result):
            return False, "Custom constraint violated"
        return True, None

opt.constraints.append(CustomConstraint())
```

## Performance Tips

1. **Start with Grid Search** for small parameter spaces (< 100 combinations)
2. **Use Bayesian (GP)** for moderate spaces (100-500 combinations) - no Optuna needed
3. **Use Optuna TPE** for large spaces (> 500 combinations) - if Optuna available
4. **Add Hard Constraints** to reduce invalid trials
5. **Monitor Convergence** to stop early if plateau reached
6. **Parallel Jobs** for CPU-bound backtests
7. **Coarse to Fine** optimization (grid search → Bayesian with refined space)

## API Reference

See `optimizer_api.py` for detailed documentation on:
- `Optimizer` class methods
- Parameter specifications
- Constraint types
- Objective function patterns

See `optimizer_examples.py` for complete code examples.

## Comparison to Original optimize.py

| Feature | optimize.py | New Optimizer |
|---------|------------|---------------|
| Strategies | Bayesian only (Optuna) | Grid, Bayesian (GP), Optuna TPE, Random, Genetic |
| Optuna Dependency | Required | Optional (only for OptunaOptimizer) |
| Default Bayesian | Optuna TPE | Gaussian Process (no deps) |
| Constraints | Basic min/max | Flexible hard/soft constraints |
| Universe Opt | Limited | Full support |
| Analysis | Basic stats | Comprehensive + sensitivity |
| Robustness | Bootstrap only | Bootstrap, Walk-forward, Monte Carlo |
| API | Functional | OOP with fluent interface |
| Extensibility | Low | High (factory pattern) |

## Migration Guide

For code using the old `optimize.py`:

```python
# Old API
from utils import optimize_universal
result = optimize_universal(train_data, val_data, strategy_class, 
                           param_space, backtest_fn, min_trades=10)

# New API
from utils import Optimizer
opt = Optimizer()
opt.add_constraint('trade_count', min_trades=10)
result = opt.optimize(train_data, val_data, strategy_class, 
                     param_space, backtest_fn)[0]
```

Both APIs are supported for backward compatibility!
"""
