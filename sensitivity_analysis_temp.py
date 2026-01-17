# Sensitivity Analysis for Best Parameters
def sensitivity_analysis(data, strategy_class, best_params, param_ranges, backtest_fn, transaction_cost=0.001):
    """
    Analyze sensitivity of strategy performance to parameter changes.
    
    Args:
        data: Price data
        strategy_class: Strategy class
        best_params: Best parameters found
        param_ranges: Dict of {param_name: (min, max, step)} to test
        backtest_fn: Backtest function
        transaction_cost: Transaction cost
    
    Returns:
        Dict with sensitivity results for each parameter
    """
    import numpy as np
    import pandas as pd
    
    results = {}
    
    # Test each parameter individually
    for param_name, (param_min, param_max, step) in param_ranges.items():
        if param_name not in best_params:
            continue
            
        test_values = list(range(param_min, param_max + 1, step))
        metrics = {
            'values': [],
            'sharpe': [],
            'total_return': [],
            'max_drawdown': [],
            'num_trades': []
        }
        
        for test_val in test_values:
            # Create params with one parameter varied
            test_params = best_params.copy()
            test_params[param_name] = test_val
            
            try:
                strategy = strategy_class(**test_params)
                backtest_result = backtest_fn(data, strategy, transaction_cost=transaction_cost)
                
                metrics['values'].append(test_val)
                metrics['sharpe'].append(backtest_result['sharpe_ratio'])
                metrics['total_return'].append(backtest_result['total_return'])
                metrics['max_drawdown'].append(backtest_result['max_drawdown'])
                metrics['num_trades'].append(backtest_result['num_trades'])
                
            except Exception as e:
                print(f"Error testing {param_name}={test_val}: {e}")
                continue
        
        results[param_name] = pd.DataFrame(metrics)
    
    return results


def plot_sensitivity(sensitivity_results, best_params, strategy_name):
    """Plot sensitivity analysis results."""
    import matplotlib.pyplot as plt
    
    n_params = len(sensitivity_results)
    fig, axes = plt.subplots(n_params, 2, figsize=(14, 4 * n_params))
    
    if n_params == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (param_name, df) in enumerate(sensitivity_results.items()):
        best_val = best_params[param_name]
        
        # Plot 1: Sharpe Ratio
        axes[idx, 0].plot(df['values'], df['sharpe'], marker='o', linewidth=2)
        axes[idx, 0].axvline(x=best_val, color='red', linestyle='--', label=f'Best: {best_val}', linewidth=2)
        axes[idx, 0].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        axes[idx, 0].set_xlabel(param_name, fontsize=11)
        axes[idx, 0].set_ylabel('Sharpe Ratio', fontsize=11)
        axes[idx, 0].set_title(f'{strategy_name} - Sharpe Sensitivity to {param_name}', fontsize=12, fontweight='bold')
        axes[idx, 0].legend()
        axes[idx, 0].grid(True, alpha=0.3)
        
        # Plot 2: Total Return
        axes[idx, 1].plot(df['values'], [x*100 for x in df['total_return']], marker='o', linewidth=2, color='green')
        axes[idx, 1].axvline(x=best_val, color='red', linestyle='--', label=f'Best: {best_val}', linewidth=2)
        axes[idx, 1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        axes[idx, 1].set_xlabel(param_name, fontsize=11)
        axes[idx, 1].set_ylabel('Total Return (%)', fontsize=11)
        axes[idx, 1].set_title(f'{strategy_name} - Return Sensitivity to {param_name}', fontsize=12, fontweight='bold')
        axes[idx, 1].legend()
        axes[idx, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def print_sensitivity_summary(sensitivity_results, best_params):
    """Print summary statistics for sensitivity analysis."""
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("="*80)
    
    for param_name, df in sensitivity_results.items():
        best_val = best_params[param_name]
        best_idx = df[df['values'] == best_val].index[0] if best_val in df['values'].values else None
        
        print(f"\n{param_name.upper()}")
        print("-" * 50)
        print(f"Best Value: {best_val}")
        
        if best_idx is not None:
            best_sharpe = df.loc[best_idx, 'sharpe']
            print(f"Best Sharpe: {best_sharpe:.4f}")
            
            # Calculate range of Sharpe within ±20% of best param
            lower_bound = int(best_val * 0.8)
            upper_bound = int(best_val * 1.2)
            nearby_df = df[(df['values'] >= lower_bound) & (df['values'] <= upper_bound)]
            
            if len(nearby_df) > 1:
                sharpe_std = nearby_df['sharpe'].std()
                sharpe_range = nearby_df['sharpe'].max() - nearby_df['sharpe'].min()
                print(f"Sharpe Std Dev (±20%): {sharpe_std:.4f}")
                print(f"Sharpe Range (±20%): {sharpe_range:.4f}")
                print(f"Sensitivity Score: {'HIGH' if sharpe_range > 0.5 else 'MEDIUM' if sharpe_range > 0.2 else 'LOW'}")
            
            # Show top 5 alternatives
            top_5 = df.nlargest(5, 'sharpe')[['values', 'sharpe', 'total_return', 'num_trades']]
            print(f"\nTop 5 Parameter Values:")
            for _, row in top_5.iterrows():
                marker = " ← BEST" if row['values'] == best_val else ""
                print(f"  {param_name}={int(row['values'])}: Sharpe={row['sharpe']:.4f}, Return={row['total_return']:.2%}, Trades={int(row['num_trades'])}{marker}")
