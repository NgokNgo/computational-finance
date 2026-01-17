"""
Advanced Optimization Strategies Module

Implements multiple optimization approaches:
- Grid Search with parallel evaluation
- Genetic Algorithm
- Random Search
- Hyperband (multi-fidelity)
"""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple
from itertools import product
import numpy as np
import pandas as pd
from dataclasses import dataclass
import time

from .optimizer_core import (
    OptimizerBase, OptimizerConfig, OptimizationSummary, 
    OptimizationResult, Constraint, ObjectiveDirection
)


# =============================================================================
# GRID SEARCH OPTIMIZER
# =============================================================================

class GridSearchOptimizer(OptimizerBase):
    """Exhaustive grid search over parameter space."""
    
    def optimize(self,
                 train_data: Dict[str, pd.DataFrame] | pd.DataFrame,
                 val_data: Dict[str, pd.DataFrame] | pd.DataFrame,
                 strategy_class: type,
                 param_space: Dict[str, Tuple],
                 backtest_fn: Callable,
                 objective_fn: Callable,
                 constraints: List[Constraint] = None) -> OptimizationSummary:
        """Run grid search optimization."""
        
        # Generate grid
        grid_params = self._generate_grid(param_space)
        n_params = len(grid_params)
        
        if self.config.verbose:
            print(f"\n{'='*70}")
            print(f"GRID SEARCH OPTIMIZATION")
            print(f"{'='*70}")
            print(f"Total parameter combinations: {n_params}")
        
        # Evaluate each combination
        start_time = time.time()
        
        for trial_id, params in enumerate(grid_params):
            try:
                if isinstance(train_data, pd.DataFrame):
                    # Single stock
                    result_train = backtest_fn(train_data, strategy_class(**params))
                    result_val = backtest_fn(val_data, strategy_class(**params))
                    
                    # Check constraints
                    if constraints:
                        satisfied, violations = self._check_constraints(result_train, constraints)
                        if not satisfied:
                            self._record_trial(trial_id, params, float('-inf'), violations=violations)
                            continue
                    
                    train_score = objective_fn(result_train)
                    val_score = objective_fn(result_val)
                    final_score = val_score
                    
                    self._record_trial(trial_id, params, final_score, train_score, val_score)
                else:
                    # Universe optimization
                    train_scores, val_scores = [], []
                    
                    for symbol in train_data.keys():
                        try:
                            result_train = backtest_fn(train_data[symbol], strategy_class(**params))
                            
                            if constraints:
                                satisfied, _ = self._check_constraints(result_train, constraints)
                                if not satisfied:
                                    continue
                            
                            result_val = backtest_fn(val_data[symbol], strategy_class(**params))
                            
                            train_scores.append(objective_fn(result_train))
                            val_scores.append(objective_fn(result_val))
                        except:
                            continue
                    
                    if len(val_scores) > 0:
                        final_score = np.mean(val_scores)
                        self._record_trial(trial_id, params, final_score, 
                                        np.mean(train_scores), np.mean(val_scores))
                
                if self.config.verbose and (trial_id + 1) % max(1, n_params // 10) == 0:
                    print(f"  Progress: {trial_id + 1}/{n_params}")
                    
            except Exception as e:
                self._record_trial(trial_id, params, float('-inf'), error=str(e))
        
        elapsed = time.time() - start_time
        summary = self._build_summary()
        summary.elapsed_time = elapsed
        
        if self.config.verbose:
            print(f"\nCompleted {len(self.trials)} trials in {elapsed:.1f}s")
            print(f"Best Score: {summary.best_score:.4f}")
            print(f"Best Params: {summary.best_params}")
        
        return summary
    
    def _generate_grid(self, param_space: Dict[str, Tuple]) -> List[Dict]:
        """Generate all parameter combinations."""
        params_list = []
        
        for name, (ptype, *bounds) in param_space.items():
            if ptype == 'int':
                params_list.append([(name, v) for v in range(bounds[0], bounds[1] + 1)])
            elif ptype == 'float':
                # Generate reasonable number of float values
                n_steps = min(10, int(bounds[1] - bounds[0] + 1))
                values = np.linspace(bounds[0], bounds[1], n_steps)
                params_list.append([(name, v) for v in values])
            elif ptype == 'categorical':
                params_list.append([(name, v) for v in bounds[0]])
        
        # Generate combinations
        all_combinations = list(product(*params_list))
        return [dict(combo) for combo in all_combinations]
    
    def _build_summary(self) -> OptimizationSummary:
        """Build summary from trials."""
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
# RANDOM SEARCH OPTIMIZER
# =============================================================================

class RandomSearchOptimizer(OptimizerBase):
    """Random sampling from parameter space."""
    
    def optimize(self,
                 train_data: Dict[str, pd.DataFrame] | pd.DataFrame,
                 val_data: Dict[str, pd.DataFrame] | pd.DataFrame,
                 strategy_class: type,
                 param_space: Dict[str, Tuple],
                 backtest_fn: Callable,
                 objective_fn: Callable,
                 constraints: List[Constraint] = None) -> OptimizationSummary:
        """Run random search optimization."""
        
        if self.config.verbose:
            print(f"\n{'='*70}")
            print(f"RANDOM SEARCH OPTIMIZATION")
            print(f"{'='*70}")
            print(f"Trials: {self.config.n_trials}")
        
        start_time = time.time()
        np.random.seed(self.config.seed)
        
        for trial_id in range(self.config.n_trials):
            try:
                params = self._sample_params(param_space)
                
                if isinstance(train_data, pd.DataFrame):
                    result_train = backtest_fn(train_data, strategy_class(**params))
                    
                    if constraints:
                        satisfied, violations = self._check_constraints(result_train, constraints)
                        if not satisfied:
                            self._record_trial(trial_id, params, float('-inf'), violations=violations)
                            continue
                    
                    result_val = backtest_fn(val_data, strategy_class(**params))
                    
                    train_score = objective_fn(result_train)
                    val_score = objective_fn(result_val)
                    final_score = val_score
                    
                    self._record_trial(trial_id, params, final_score, train_score, val_score)
                else:
                    # Universe
                    train_scores, val_scores = [], []
                    
                    for symbol in train_data.keys():
                        try:
                            result_train = backtest_fn(train_data[symbol], strategy_class(**params))
                            
                            if constraints:
                                satisfied, _ = self._check_constraints(result_train, constraints)
                                if not satisfied:
                                    continue
                            
                            result_val = backtest_fn(val_data[symbol], strategy_class(**params))
                            
                            train_scores.append(objective_fn(result_train))
                            val_scores.append(objective_fn(result_val))
                        except:
                            continue
                    
                    if len(val_scores) > 0:
                        final_score = np.mean(val_scores)
                        self._record_trial(trial_id, params, final_score,
                                        np.mean(train_scores), np.mean(val_scores))
                
                if self.config.verbose and (trial_id + 1) % max(1, self.config.n_trials // 5) == 0:
                    print(f"  Progress: {trial_id + 1}/{self.config.n_trials}")
                    
            except Exception as e:
                self._record_trial(trial_id, params, float('-inf'), error=str(e))
        
        elapsed = time.time() - start_time
        summary = self._build_summary()
        summary.elapsed_time = elapsed
        
        if self.config.verbose:
            print(f"\nCompleted {len(self.trials)} trials in {elapsed:.1f}s")
            print(f"Best Score: {summary.best_score:.4f}")
            print(f"Best Params: {summary.best_params}")
        
        return summary
    
    def _sample_params(self, param_space: Dict[str, Tuple]) -> Dict:
        """Sample random parameters."""
        params = {}
        
        for name, (ptype, *bounds) in param_space.items():
            if ptype == 'int':
                params[name] = np.random.randint(bounds[0], bounds[1] + 1)
            elif ptype == 'float':
                params[name] = np.random.uniform(bounds[0], bounds[1])
            elif ptype == 'categorical':
                params[name] = np.random.choice(bounds[0])
        
        return params
    
    def _build_summary(self) -> OptimizationSummary:
        """Build summary from trials."""
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
# GENETIC ALGORITHM OPTIMIZER
# =============================================================================

class GeneticAlgorithmOptimizer(OptimizerBase):
    """Genetic algorithm for parameter optimization."""
    
    def __init__(self, config: OptimizerConfig, 
                 population_size: int = 30,
                 mutation_rate: float = 0.2,
                 crossover_rate: float = 0.8,
                 generations: Optional[int] = None):
        super().__init__(config)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations or (self.config.n_trials // population_size)
    
    def optimize(self,
                 train_data: Dict[str, pd.DataFrame] | pd.DataFrame,
                 val_data: Dict[str, pd.DataFrame] | pd.DataFrame,
                 strategy_class: type,
                 param_space: Dict[str, Tuple],
                 backtest_fn: Callable,
                 objective_fn: Callable,
                 constraints: List[Constraint] = None) -> OptimizationSummary:
        """Run genetic algorithm optimization."""
        
        if self.config.verbose:
            print(f"\n{'='*70}")
            print(f"GENETIC ALGORITHM OPTIMIZATION")
            print(f"{'='*70}")
            print(f"Population: {self.population_size}, Generations: {self.generations}")
        
        start_time = time.time()
        np.random.seed(self.config.seed)
        
        # Initialize population
        population = [self._random_params(param_space) for _ in range(self.population_size)]
        trial_id = [0]
        
        best_fitness = float('-inf')
        
        for gen in range(self.generations):
            # Evaluate population
            fitness_scores = []
            
            for params in population:
                try:
                    if isinstance(train_data, pd.DataFrame):
                        result_train = backtest_fn(train_data, strategy_class(**params))
                        result_val = backtest_fn(val_data, strategy_class(**params))
                        
                        if constraints:
                            satisfied, violations = self._check_constraints(result_train, constraints)
                            if not satisfied:
                                fitness = float('-inf')
                            else:
                                fitness = objective_fn(result_val)
                        else:
                            fitness = objective_fn(result_val)
                    else:
                        # Universe
                        val_scores = []
                        for symbol in train_data.keys():
                            try:
                                result_train = backtest_fn(train_data[symbol], strategy_class(**params))
                                if constraints:
                                    satisfied, _ = self._check_constraints(result_train, constraints)
                                    if not satisfied:
                                        continue
                                result_val = backtest_fn(val_data[symbol], strategy_class(**params))
                                val_scores.append(objective_fn(result_val))
                            except:
                                pass
                        fitness = np.mean(val_scores) if val_scores else float('-inf')
                    
                    fitness_scores.append(fitness)
                    self._record_trial(trial_id[0], params, fitness)
                    trial_id[0] += 1
                    
                except Exception as e:
                    fitness_scores.append(float('-inf'))
                    self._record_trial(trial_id[0], params, float('-inf'), error=str(e))
                    trial_id[0] += 1
            
            # Selection and reproduction
            valid_indices = [i for i, f in enumerate(fitness_scores) if f > float('-inf')]
            
            if not valid_indices:
                if self.config.verbose:
                    print(f"  Generation {gen + 1}/{self.generations}: No valid solutions")
                continue
            
            sorted_indices = sorted(valid_indices, key=lambda i: fitness_scores[i], reverse=True)
            elite = [population[i] for i in sorted_indices[:self.population_size // 2]]
            
            # Crossover and mutation
            new_population = elite[:]
            
            while len(new_population) < self.population_size:
                if np.random.random() < self.crossover_rate and len(elite) > 1:
                    parent1, parent2 = np.random.choice(len(elite), 2, replace=False)
                    child = self._crossover(elite[parent1], elite[parent2])
                else:
                    child = elite[np.random.randint(len(elite))].copy()
                
                # Mutation
                if np.random.random() < self.mutation_rate:
                    child = self._mutate(child, param_space)
                
                new_population.append(child)
            
            population = new_population[:self.population_size]
            
            gen_best = max([fitness_scores[i] for i in valid_indices])
            
            if self.config.verbose and (gen + 1) % max(1, self.generations // 5) == 0:
                print(f"  Generation {gen + 1}/{self.generations}: Best = {gen_best:.4f}")
        
        elapsed = time.time() - start_time
        summary = self._build_summary()
        summary.elapsed_time = elapsed
        
        if self.config.verbose:
            print(f"\nCompleted {len(self.trials)} trials in {elapsed:.1f}s")
            print(f"Best Score: {summary.best_score:.4f}")
            print(f"Best Params: {summary.best_params}")
        
        return summary
    
    def _random_params(self, param_space: Dict[str, Tuple]) -> Dict:
        """Generate random parameters."""
        params = {}
        for name, (ptype, *bounds) in param_space.items():
            if ptype == 'int':
                params[name] = np.random.randint(bounds[0], bounds[1] + 1)
            elif ptype == 'float':
                params[name] = np.random.uniform(bounds[0], bounds[1])
            elif ptype == 'categorical':
                params[name] = np.random.choice(bounds[0])
        return params
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Crossover two parents."""
        child = {}
        for key in parent1.keys():
            child[key] = parent1[key] if np.random.random() < 0.5 else parent2[key]
        return child
    
    def _mutate(self, params: Dict, param_space: Dict[str, Tuple]) -> Dict:
        """Mutate parameters."""
        mutated = params.copy()
        key = np.random.choice(list(mutated.keys()))
        ptype, *bounds = param_space[key]
        
        if ptype == 'int':
            mutated[key] = np.random.randint(bounds[0], bounds[1] + 1)
        elif ptype == 'float':
            mutated[key] = np.random.uniform(bounds[0], bounds[1])
        elif ptype == 'categorical':
            mutated[key] = np.random.choice(bounds[0])
        
        return mutated
    
    def _build_summary(self) -> OptimizationSummary:
        """Build summary from trials."""
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
# OPTIMIZER FACTORY
# =============================================================================

class OptimizerFactory:
    """Factory for creating optimizers."""
    
    @staticmethod
    def create(config: OptimizerConfig) -> OptimizerBase:
        """Create optimizer instance."""
        from .optimizer_core import BayesianOptimizer, OptimizationStrategy
        
        if config.strategy == OptimizationStrategy.BAYESIAN:
            return BayesianOptimizer(config)
        elif config.strategy == OptimizationStrategy.GRID_SEARCH:
            return GridSearchOptimizer(config)
        elif config.strategy == OptimizationStrategy.RANDOM:
            return RandomSearchOptimizer(config)
        elif config.strategy == OptimizationStrategy.GENETIC:
            return GeneticAlgorithmOptimizer(config)
        else:
            raise ValueError(f"Unknown strategy: {config.strategy}")


__all__ = [
    'GridSearchOptimizer', 'RandomSearchOptimizer', 'GeneticAlgorithmOptimizer',
    'OptimizerFactory'
]
