"""
LEAN-INSPIRED OPTIMIZER - ARCHITECTURE DIAGRAM

Visual representation of the modular optimizer architecture.
"""

ARCHITECTURE LAYERS
===================

┌─────────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                               │
│                    (User Code)                                      │
│                                                                     │
│  my_optimization_script.py:                                        │
│    opt = Optimizer()                                              │
│    opt.add_constraint(...).set_objective(...)                     │
│    result = opt.optimize(...)                                     │
└────────────────┬────────────────────────────────────────────────────┘
                 │
┌────────────────┴────────────────────────────────────────────────────┐
│               HIGH-LEVEL API LAYER                                 │
│               (optimizer_api.py)                                   │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Optimizer                                                   │ │
│  │ - fluent interface                                          │ │
│  │ - add_constraint()                                          │ │
│  │ - set_objective()                                           │ │
│  │ - optimize()                                                │ │
│  │ - test_robustness()                                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ quick_optimize()  [convenience function]                   │ │
│  └─────────────────────────────────────────────────────────────┘ │
└────────────────┬────────────────────────────────────────────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
┌───▼──────────────────┐ ┌───▼──────────────────┐
│ STRATEGY LAYER       │ │ ANALYSIS LAYER       │
│ (optimizer_...)      │ │ (optimizer_...)      │
└────────────┬─────────┘ └────────────┬─────────┘
             │                        │
             │                        │
    ┌────────▼────────┐      ┌────────▼──────────┐
    │ Strategies      │      │ Analysis          │
    │ (Core Logic)    │      │ (Post-opt)        │
    │                 │      │                   │
    │ • Bayesian      │      │ • ResultsAnalyzer │
    │ • GridSearch    │      │ • ParamSensitiv.. │
    │ • Random        │      │ • RobustnessTstr  │
    │ • Genetic       │      │                   │
    └────────┬────────┘      └────────┬──────────┘
             │                        │
└────────────┴────────────┬───────────┘
                          │
        ┌─────────────────▼────────────────────┐
        │  CORE FRAMEWORK LAYER                │
        │  (optimizer_core.py)                 │
        │                                      │
        │  ┌──────────────────────────────┐   │
        │  │ OptimizerBase (ABC)          │   │
        │  │ - abstract optimize()        │   │
        │  │ - _check_constraints()       │   │
        │  │ - _record_trial()            │   │
        │  └──────────────────────────────┘   │
        │                                      │
        │  ┌──────────────────────────────┐   │
        │  │ Enums & Models               │   │
        │  │ - OptimizationStrategy       │   │
        │  │ - ObjectiveDirection         │   │
        │  │ - ConstraintType             │   │
        │  │ - OptimizationResult         │   │
        │  │ - OptimizationSummary        │   │
        │  │ - OptimizerConfig            │   │
        │  └──────────────────────────────┘   │
        │                                      │
        │  ┌──────────────────────────────┐   │
        │  │ Constraint System            │   │
        │  │ - Constraint (base)          │   │
        │  │ - TradeCountConstraint       │   │
        │  │ - SharpeConstraint           │   │
        │  │ - DrawdownConstraint         │   │
        │  │ - ReturnConstraint           │   │
        │  └──────────────────────────────┘   │
        │                                      │
        │  ┌──────────────────────────────┐   │
        │  │ ObjectiveBuilder             │   │
        │  │ - add_component()            │   │
        │  │ - add_penalty()              │   │
        │  │ - build()                    │   │
        │  └──────────────────────────────┘   │
        │                                      │
        └──────────────────────────────────────┘
             │
             │
        ┌────▼────────────────────────────────┐
        │  EXECUTION LAYER                     │
        │  (External Libraries)                │
        │                                      │
        │  • optuna (Bayesian)                │
        │  • pandas (data handling)           │
        │  • numpy (computation)              │
        │  • User's backtest_fn               │
        │  • User's strategy_class            │
        └────────────────────────────────────┘


COMPONENT RESPONSIBILITIES
==========================

┌─────────────────────────────────────────────────────────────────┐
│ optimizer_core.py (Foundation Layer)                            │
├─────────────────────────────────────────────────────────────────┤
│ • Data structures (Result, Summary, Config)                     │
│ • Constraint framework (definition and checking)                │
│ • Objective building (composition and penalties)                │
│ • Abstract optimizer base class                                 │
│ • Bayesian optimizer implementation (Optuna)                    │
│                                                                 │
│ Exports: 13 classes, 3 enums                                   │
│ ~445 lines                                                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ optimizer_strategies.py (Strategy Layer)                        │
├─────────────────────────────────────────────────────────────────┤
│ • Grid Search optimizer                                         │
│ • Random Search optimizer                                       │
│ • Genetic Algorithm optimizer                                   │
│ • Optimizer factory (creation pattern)                          │
│                                                                 │
│ Exports: 4 classes                                             │
│ ~425 lines                                                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ optimizer_analyzer.py (Analysis Layer)                          │
├─────────────────────────────────────────────────────────────────┤
│ • Results analysis (statistics, top params, convergence)        │
│ • Parameter sensitivity analysis (single and interactions)      │
│ • Robustness testing (bootstrap, walk-forward, Monte Carlo)     │
│ • Report generation                                             │
│                                                                 │
│ Exports: 3 classes + nested models                             │
│ ~285 lines                                                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ optimizer_api.py (User API Layer)                               │
├─────────────────────────────────────────────────────────────────┤
│ • Optimizer class (fluent interface)                            │
│ • quick_optimize function                                       │
│ • High-level convenience methods                                │
│                                                                 │
│ Exports: 2 main classes + enums                                │
│ ~365 lines                                                      │
└─────────────────────────────────────────────────────────────────┘


OPTIMIZATION WORKFLOW
====================

User Code
   │
   ├─> Create Optimizer instance
   │      │
   │      └─> OptimizerConfig created
   │
   ├─> Add Constraints (chainable)
   │      │
   │      └─> Constraint instances appended to list
   │
   ├─> Set Objective (chainable)
   │      │
   │      └─> ObjectiveBuilder creates Callable
   │
   ├─> Call optimize()
   │      │
   │      ├─> OptimizerFactory.create() 
   │      │      │
   │      │      └─> Returns optimizer instance (Bayesian/Grid/etc)
   │      │
   │      ├─> optimizer.optimize() executes
   │      │      │
   │      │      ├─> For each trial:
   │      │      │    ├─> Sample parameters
   │      │      │    ├─> Backtest strategy
   │      │      │    ├─> Check hard constraints (reject if fail)
   │      │      │    ├─> Calculate objective score
   │      │      │    ├─> Apply soft penalties
   │      │      │    └─> Record OptimizationResult
   │      │      │
   │      │      └─> Return OptimizationSummary
   │      │
   │      └─> ResultsAnalyzer initialized
   │
   ├─> Analyze results
   │      │
   │      ├─> analyzer.generate_report()
   │      ├─> analyzer.analyze_parameter_sensitivity()
   │      ├─> analyzer.analyze_parameter_interactions()
   │      └─> analyzer.get_overfitting_analysis()
   │
   └─> Test robustness
          │
          ├─> Bootstrap test
          ├─> Walk-forward test
          └─> Monte Carlo test


CONSTRAINT EVALUATION FLOW
==========================

Trial Execution
   │
   ├─> Sample Parameters
   │
   ├─> Backtest Strategy
   │
   ├─> Check Hard Constraints
   │   │
   │   ├─> constraint.check(result)
   │   │
   │   ├─> If violated → Return -inf, skip trial
   │   │
   │   └─> If satisfied → Continue
   │
   ├─> Calculate Objective Score
   │
   ├─> Apply Soft Penalty Constraints
   │   │
   │   ├─> For each soft constraint:
   │   │   ├─> Calculate penalty
   │   │   └─> Reduce score by penalty amount
   │   │
   │   └─> Final Score = Objective - Soft Penalties
   │
   └─> Record OptimizationResult


OBJECTIVE FUNCTION COMPOSITION
==============================

User specifies via set_objective():
   │
   ├─> Main Components (weights):
   │   ├─> Sharpe Ratio (0.4)
   │   ├─> Return (0.3)
   │   └─> Drawdown (-0.2)
   │
   ├─> Penalties (weights):
   │   ├─> Overfit (0.3)
   │   ├─> Consistency (0.1)
   │   └─> Trade Variance (0.0)
   │
   └─> Final Score = sum(components) - sum(penalties)


DATA FLOW: Single Stock Optimization
====================================

Training Data          Validation Data
    │                       │
    └───────┬───────────────┘
            │
            ├─> Optuna Trial
            │   │
            │   ├─> Sample Parameters
            │   │
            │   ├─> Backtest(train_data, params)
            │   │   └─> Returns: num_trades, sharpe, return, drawdown, ...
            │   │
            │   ├─> Check Constraints(train_result)
            │   │
            │   ├─> Calculate train_score = objective_fn(train_result)
            │   │
            │   ├─> Backtest(val_data, params)
            │   │   └─> Returns: num_trades, sharpe, return, drawdown, ...
            │   │
            │   ├─> Calculate val_score = objective_fn(val_result)
            │   │
            │   ├─> Apply Penalties
            │   │   ├─> Overfit: train_score - val_score
            │   │   └─> Consistency: std(scores) across prior trials
            │   │
            │   └─> final_score = val_score - penalties
            │
            ├─> Record OptimizationResult
            │   └─> {params, score, train_score, val_score, metrics}
            │
            └─> Optuna selects next parameters based on history


DATA FLOW: Universe Optimization
================================

All Stock Data (dict of DataFrames)
    │
    ├─> For Each Stock:
    │   │
    │   ├─> Backtest(train_data[symbol], params)
    │   │
    │   ├─> Check Constraints(result)
    │   │   ├─> If hard constraint fails → skip stock
    │   │   └─> If passed → collect score
    │   │
    │   ├─> Backtest(val_data[symbol], params)
    │   │
    │   ├─> Calculate val_score
    │   │
    │   └─> Record per_stock_results[symbol]
    │
    ├─> Aggregate Results
    │   ├─> avg_val_score = mean(all val_scores)
    │   ├─> consistency = std(all val_scores)
    │   └─> Apply penalties to avg_val_score
    │
    └─> final_score = avg_val_score - penalties


EXTENSIBILITY POINTS
====================

Add New Optimization Strategy:
    1. Create subclass of OptimizerBase
    2. Implement optimize() method
    3. Register in OptimizerFactory
    Done! Available via Optimizer(strategy=NEW_STRATEGY)

Add New Constraint Type:
    1. Create subclass of Constraint
    2. Implement check() method
    3. Use via opt.add_constraint('custom_name', ...)
    Done! Works with hard/soft types

Add New Objective Component:
    1. Define custom function: lambda r: ...
    2. Use ObjectiveBuilder.add_component()
    Done! Composable objective ready

Add New Analysis Method:
    1. Add method to ResultsAnalyzer class
    2. Called on results after optimization
    Done! Available via analyzer.your_method()


KEY DESIGN DECISIONS
====================

1. OOP over Functional
   - Better encapsulation
   - Easier extension
   - More maintainable

2. Fluent Interface
   - Chainable methods
   - Readable code
   - Similar to SQL query builders

3. Factory Pattern
   - Flexible strategy selection
   - Easy to add new strategies
   - No tight coupling

4. Separation of Concerns
   - Core framework separate from strategies
   - Analysis separate from optimization
   - API layer separate from implementation

5. Data Classes for Results
   - Structured result representation
   - Easy serialization
   - Type safety

6. Constraint-based Validation
   - Flexible constraint system
   - Hard and soft constraints
   - Extensible for custom constraints

7. Penalty-based Objectives
   - Avoid hard-coded scoring logic
   - Composable penalties
   - Transparent scoring


SUMMARY
=======

A sophisticated, modular, enterprise-grade optimizer framework
that brings QuantConnect/Lean design principles to parameter
optimization for trading strategy development.

Levels:
  ├─ Application Layer (User Code)
  ├─ High-Level API (optimizer_api.py)
  ├─ Strategy & Analysis (optimizer_strategies.py, optimizer_analyzer.py)
  ├─ Core Framework (optimizer_core.py)
  └─ External Libraries (Optuna, pandas, numpy)

Features:
  ✓ 4 optimization strategies
  ✓ Flexible constraints
  ✓ Custom objectives
  ✓ Comprehensive analysis
  ✓ Robustness testing
  ✓ Universe optimization
  ✓ Production-ready
  ✓ Well-documented
"""
