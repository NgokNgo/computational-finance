"""
Optimization Results Analysis and Reporting Module

Provides comprehensive analysis of optimization results:
- Statistical summaries
- Parameter sensitivity analysis
- Robustness testing
- Results visualization preparation
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .optimizer_core import OptimizationResult, OptimizationSummary


# =============================================================================
# RESULT ANALYZERS
# =============================================================================

@dataclass
class ParameterSensitivity:
    """Parameter sensitivity analysis."""
    parameter: str
    values: List[float]
    scores: List[float]
    optimal_value: float
    correlation: float


class ResultsAnalyzer:
    """Comprehensive analysis of optimization results."""
    
    def __init__(self, trials: List[OptimizationResult], summary: OptimizationSummary):
        self.trials = trials
        self.summary = summary
        self.valid_trials = [t for t in trials if t.is_valid()]
    
    def get_statistics(self) -> Dict[str, float]:
        """Get comprehensive statistics."""
        scores = [t.score for t in self.valid_trials]
        
        if not scores:
            return {}
        
        return {
            'mean': np.mean(scores),
            'median': np.median(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'q25': np.percentile(scores, 25),
            'q75': np.percentile(scores, 75),
            'range': np.max(scores) - np.min(scores),
            'cv': np.std(scores) / np.mean(scores) if np.mean(scores) != 0 else np.inf
        }
    
    def get_best_parameters(self, n: int = 5) -> List[Tuple[Dict, float]]:
        """Get top N best parameter sets."""
        sorted_trials = sorted(self.valid_trials, key=lambda t: t.score, reverse=True)
        return [(t.params, t.score) for t in sorted_trials[:n]]
    
    def analyze_parameter_sensitivity(self, param_name: str) -> ParameterSensitivity:
        """Analyze sensitivity to single parameter."""
        values = []
        scores = []
        
        for trial in self.valid_trials:
            if param_name in trial.params:
                values.append(trial.params[param_name])
                scores.append(trial.score)
        
        if not values:
            raise ValueError(f"Parameter {param_name} not found in trials")
        
        # Sort by parameter value
        sorted_pairs = sorted(zip(values, scores), key=lambda x: x[0])
        values, scores = zip(*sorted_pairs)
        values, scores = list(values), list(scores)
        
        # Find optimal
        optimal_idx = np.argmax(scores)
        
        # Calculate correlation
        if len(set(values)) > 1:
            correlation = np.corrcoef(values, scores)[0, 1]
        else:
            correlation = 0.0
        
        return ParameterSensitivity(
            parameter=param_name,
            values=values,
            scores=scores,
            optimal_value=values[optimal_idx],
            correlation=correlation
        )
    
    def analyze_parameter_interactions(self, param1: str, param2: str) -> pd.DataFrame:
        """Analyze interaction between two parameters."""
        data = []
        
        for trial in self.valid_trials:
            if param1 in trial.params and param2 in trial.params:
                data.append({
                    param1: trial.params[param1],
                    param2: trial.params[param2],
                    'score': trial.score
                })
        
        df = pd.DataFrame(data)
        
        if df.empty:
            return df
        
        # Create pivot table
        pivot = df.pivot_table(
            values='score',
            index=param1,
            columns=param2,
            aggfunc='mean'
        )
        
        return pivot
    
    def get_overfitting_analysis(self) -> Dict:
        """Analyze overfitting in results."""
        overfit_gaps = []
        
        for trial in self.valid_trials:
            if trial.train_score is not None and trial.val_score is not None:
                gap = trial.train_score - trial.val_score
                overfit_gaps.append(gap)
        
        if not overfit_gaps:
            return {}
        
        return {
            'mean_gap': np.mean(overfit_gaps),
            'max_gap': np.max(overfit_gaps),
            'min_gap': np.min(overfit_gaps),
            'std_gap': np.std(overfit_gaps),
            'trials_with_overfitting': sum(1 for g in overfit_gaps if g > 0.1),
            'overfitting_ratio': sum(1 for g in overfit_gaps if g > 0.1) / len(overfit_gaps)
        }
    
    def get_constraint_analysis(self) -> Dict:
        """Analyze constraint violations."""
        violations_summary = {}
        
        for trial in self.trials:
            for violation in trial.constraints_violated:
                violations_summary[violation] = violations_summary.get(violation, 0) + 1
        
        return {
            'total_violations': len(self.trials) - len(self.valid_trials),
            'violation_details': violations_summary,
            'valid_ratio': len(self.valid_trials) / len(self.trials) if self.trials else 0
        }
    
    def get_convergence_analysis(self) -> Dict:
        """Analyze optimization convergence."""
        if len(self.valid_trials) < 2:
            return {}
        
        best_scores = []
        current_best = float('-inf')
        
        for trial in sorted(self.valid_trials, key=lambda t: t.trial_id):
            current_best = max(current_best, trial.score)
            best_scores.append(current_best)
        
        improvements = [best_scores[i] - best_scores[i-1] for i in range(1, len(best_scores))]
        improvements = [imp for imp in improvements if imp > 0]
        
        return {
            'n_improvements': len(improvements),
            'total_improvement': best_scores[-1] - best_scores[0] if best_scores else 0,
            'mean_improvement': np.mean(improvements) if improvements else 0,
            'improvement_rate': len(improvements) / len(best_scores) if best_scores else 0,
            'best_scores': best_scores
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive text report."""
        report = []
        report.append("=" * 80)
        report.append("OPTIMIZATION ANALYSIS REPORT")
        report.append("=" * 80)
        
        # Summary
        report.append("\n📊 OPTIMIZATION SUMMARY")
        report.append(f"  Best Score:     {self.summary.best_score:.6f}")
        report.append(f"  Best Trial ID:  {self.summary.best_trial_id}")
        report.append(f"  Total Trials:   {self.summary.n_trials}")
        report.append(f"  Valid Trials:   {self.summary.n_valid_trials}")
        report.append(f"  Success Rate:   {100 * self.summary.n_valid_trials / self.summary.n_trials:.1f}%")
        
        # Statistics
        stats = self.get_statistics()
        if stats:
            report.append("\n📈 SCORE STATISTICS")
            report.append(f"  Mean:           {stats['mean']:.6f}")
            report.append(f"  Median:         {stats['median']:.6f}")
            report.append(f"  Std Dev:        {stats['std']:.6f}")
            report.append(f"  Range:          [{stats['min']:.6f}, {stats['max']:.6f}]")
            report.append(f"  IQR:            [{stats['q25']:.6f}, {stats['q75']:.6f}]")
            report.append(f"  Coef. of Var.:  {stats['cv']:.4f}")
        
        # Best parameters
        report.append("\n🎯 TOP 5 PARAMETER SETS")
        for i, (params, score) in enumerate(self.get_best_parameters(5), 1):
            report.append(f"  #{i} (Score: {score:.6f})")
            for k, v in params.items():
                report.append(f"      {k}: {v}")
        
        # Overfitting analysis
        overfit = self.get_overfitting_analysis()
        if overfit:
            report.append("\n⚠️  OVERFITTING ANALYSIS")
            report.append(f"  Mean Gap:              {overfit['mean_gap']:.6f}")
            report.append(f"  Max Gap:               {overfit['max_gap']:.6f}")
            report.append(f"  Overfitting Ratio:     {100 * overfit['overfitting_ratio']:.1f}%")
            
            if overfit['overfitting_ratio'] < 0.2:
                report.append("  ✅ Good generalization")
            elif overfit['overfitting_ratio'] < 0.5:
                report.append("  ⚠️ Moderate overfitting detected")
            else:
                report.append("  ❌ Significant overfitting detected")
        
        # Constraints
        constraints = self.get_constraint_analysis()
        if constraints['total_violations'] > 0:
            report.append("\n🚫 CONSTRAINT VIOLATIONS")
            report.append(f"  Total Violations:  {constraints['total_violations']}")
            for violation, count in constraints['violation_details'].items():
                report.append(f"    - {violation}: {count}")
        
        # Convergence
        convergence = self.get_convergence_analysis()
        if convergence:
            report.append("\n📉 CONVERGENCE ANALYSIS")
            report.append(f"  Improvements:        {convergence['n_improvements']}")
            report.append(f"  Total Improvement:   {convergence['total_improvement']:.6f}")
            report.append(f"  Improvement Rate:    {100 * convergence['improvement_rate']:.1f}%")
        
        # Timing
        if self.summary.elapsed_time:
            report.append("\n⏱️  TIMING")
            report.append(f"  Total Time:    {self.summary.elapsed_time:.1f}s")
            if self.summary.n_trials > 0:
                report.append(f"  Time per Trial: {self.summary.elapsed_time / self.summary.n_trials:.3f}s")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


# =============================================================================
# ROBUSTNESS TESTER
# =============================================================================

class RobustnessTester:
    """Test robustness of optimized parameters."""
    
    @staticmethod
    def bootstrap_test(df: pd.DataFrame,
                      strategy_class: type,
                      params: Dict,
                      backtest_fn,
                      n_bootstrap: int = 100,
                      sample_ratio: float = 0.8) -> Dict:
        """Bootstrap resampling test."""
        scores = []
        
        for _ in range(n_bootstrap):
            indices = np.sort(np.random.choice(len(df), int(len(df) * sample_ratio), replace=True))
            try:
                result = backtest_fn(df.iloc[indices].reset_index(drop=True), strategy_class(**params))
                scores.append(result['sharpe_ratio'])
            except:
                continue
        
        if not scores:
            return {}
        
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'ci_lower': np.percentile(scores, 2.5),
            'ci_upper': np.percentile(scores, 97.5),
            'min': np.min(scores),
            'max': np.max(scores),
            'n_success': len(scores)
        }
    
    @staticmethod
    def walk_forward_test(df: pd.DataFrame,
                         strategy_class: type,
                         params: Dict,
                         backtest_fn,
                         n_periods: int = 5) -> Dict:
        """Walk-forward out-of-sample test."""
        df = df.sort_values('date').reset_index(drop=True)
        period_size = len(df) // n_periods
        period_scores = []
        
        for i in range(n_periods):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size
            test_df = df.iloc[start_idx:end_idx]
            
            try:
                result = backtest_fn(test_df, strategy_class(**params))
                period_scores.append(result['sharpe_ratio'])
            except:
                continue
        
        if not period_scores:
            return {}
        
        return {
            'period_scores': period_scores,
            'mean': np.mean(period_scores),
            'std': np.std(period_scores),
            'min': np.min(period_scores),
            'max': np.max(period_scores),
            'consistency': 1 - (np.std(period_scores) / np.mean(period_scores)) if np.mean(period_scores) > 0 else 0
        }
    
    @staticmethod
    def monte_carlo_test(df: pd.DataFrame,
                        strategy_class: type,
                        params: Dict,
                        backtest_fn,
                        n_simulations: int = 100) -> Dict:
        """Monte Carlo simulation with order randomization."""
        scores = []
        
        for _ in range(n_simulations):
            shuffled_idx = np.random.permutation(len(df))
            shuffled_df = df.iloc[shuffled_idx].reset_index(drop=True)
            
            try:
                result = backtest_fn(shuffled_df, strategy_class(**params))
                scores.append(result['sharpe_ratio'])
            except:
                continue
        
        if not scores:
            return {}
        
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'percentile_5': np.percentile(scores, 5),
            'percentile_95': np.percentile(scores, 95),
            'n_success': len(scores),
            'success_rate': len(scores) / n_simulations
        }


__all__ = [
    'ParameterSensitivity',
    'ResultsAnalyzer',
    'RobustnessTester'
]
