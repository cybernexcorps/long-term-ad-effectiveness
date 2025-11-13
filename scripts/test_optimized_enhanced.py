#!/usr/bin/env python
"""
Enhanced test with improved convergence, posterior predictive checks, and budget optimization.

Improvements over test_optimized.py:
1. 4 chains × 500 draws for robust convergence (R-hat < 1.01)
2. Posterior predictive checks to validate model fit
3. Budget optimization using ROI estimates
4. JAX backend for 5-20x speedup

Usage:
    python scripts/test_optimized_enhanced.py
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import arviz as az
from scipy.optimize import minimize

# Note: JAX is disabled for this test due to incompatibility with multiprocessing
# JAX + os.fork() causes deadlocks when using multiple chains
# For single-chain runs, you can enable JAX for 5-20x speedup
print("ℹ JAX backend DISABLED (multiprocessing incompatibility)")
print("  Using default PyTensor backend for stable 4-chain sampling")

from mmm_optimized import UCM_MMM_Optimized
from bvar_optimized import BVAR_Optimized


def posterior_predictive_check(mmm, actual_sales, n_samples=100):
    """
    Perform posterior predictive check to validate model fit.

    Samples from the posterior and generates predicted sales to compare
    with actual observed sales.

    Args:
        mmm: Fitted UCM_MMM_Optimized model
        actual_sales: Actual observed sales data
        n_samples: Number of posterior samples to use

    Returns:
        dict with diagnostics and plot data
    """
    print("\n   --- Posterior Predictive Check ---")
    print(f"   → Generating {n_samples} posterior predictive samples...")

    # Generate posterior predictive samples
    with mmm.model:
        ppc = az.sample_posterior_predictive(
            mmm.trace,
            var_names=['y_obs'],
            random_seed=42
        )

    # Extract predictions
    y_pred = ppc.posterior_predictive['y_obs'].values  # Shape: (chains, draws, time)

    # Flatten across chains and draws, then sample
    y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
    sample_indices = np.random.choice(y_pred_flat.shape[0], size=min(n_samples, y_pred_flat.shape[0]), replace=False)
    y_pred_samples = y_pred_flat[sample_indices]

    # Calculate summary statistics
    y_pred_mean = y_pred_samples.mean(axis=0)
    y_pred_lower = np.percentile(y_pred_samples, 2.5, axis=0)
    y_pred_upper = np.percentile(y_pred_samples, 97.5, axis=0)

    # Calculate fit metrics
    residuals = actual_sales - y_pred_mean
    mape = np.mean(np.abs(residuals / actual_sales)) * 100
    r2 = 1 - np.sum(residuals**2) / np.sum((actual_sales - actual_sales.mean())**2)

    # Check if actual sales fall within credible intervals
    coverage = np.mean((actual_sales >= y_pred_lower) & (actual_sales <= y_pred_upper)) * 100

    print(f"   ✓ Posterior predictive check completed")
    print(f"   → MAPE: {mape:.2f}% (lower is better)")
    print(f"   → R²: {r2:.3f} (higher is better)")
    print(f"   → 95% CI Coverage: {coverage:.1f}% (should be ~95%)")

    if mape < 10:
        print("   ✓ Excellent fit (MAPE < 10%)")
    elif mape < 20:
        print("   ✓ Good fit (MAPE < 20%)")
    else:
        print("   ⚠ Model may need improvement (MAPE > 20%)")

    return {
        'y_pred_mean': y_pred_mean,
        'y_pred_lower': y_pred_lower,
        'y_pred_upper': y_pred_upper,
        'mape': mape,
        'r2': r2,
        'coverage': coverage
    }


def optimize_budget(total_roi, current_spend, total_budget):
    """
    Optimize budget allocation across channels to maximize total ROI.

    Uses scipy.optimize to find optimal spend allocation given:
    - Current ROI per channel
    - Current spend per channel
    - Total budget constraint
    - Diminishing returns (ROI decreases with spend)

    Args:
        total_roi: Dict of channel -> total ROI per $1
        current_spend: Dict of channel -> current average spend
        total_budget: Total marketing budget to allocate

    Returns:
        dict with optimal allocation and expected return
    """
    print("\n   --- Budget Optimization ---")
    print(f"   → Total budget: ${total_budget:,.0f}")
    print(f"   → Optimizing allocation across {len(total_roi)} channels...")

    channels = list(total_roi.keys())
    current_roi = np.array([total_roi[ch] for ch in channels])
    current_spend_arr = np.array([current_spend[ch] for ch in channels])

    # Objective: Maximize expected return accounting for diminishing returns
    # Model: return = roi * spend^0.7 (power < 1 = diminishing returns)
    def negative_return(spend):
        # Use power function to model diminishing returns
        return -np.sum(current_roi * np.power(spend, 0.7))

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - total_budget}  # Spend exactly total_budget
    ]

    # Bounds: each channel gets at least 5% of budget, at most 60%
    bounds = [(total_budget * 0.05, total_budget * 0.60) for _ in channels]

    # Initial guess: proportional to current ROI
    roi_weights = current_roi / current_roi.sum()
    x0 = roi_weights * total_budget

    # Optimize
    result = minimize(
        negative_return,
        x0=x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )

    if not result.success:
        print(f"   ⚠ Optimization warning: {result.message}")

    optimal_spend = result.x
    expected_return = -result.fun

    print(f"   ✓ Optimization completed")
    print(f"   → Expected return: ${expected_return:,.0f}")
    print(f"   → Expected ROI: ${expected_return/total_budget:.2f} per $1")

    print("\n   --- Optimal Allocation vs Current ---")
    print(f"   {'Channel':<20s} {'Current':<12s} {'Optimal':<12s} {'Change':<12s}")
    print("   " + "-"*60)

    results = {}
    for i, channel in enumerate(channels):
        current = current_spend_arr[i]
        optimal = optimal_spend[i]
        change_pct = (optimal - current) / current * 100 if current > 0 else 0

        print(f"   {channel:<20s} ${current:>10,.0f} ${optimal:>10,.0f} {change_pct:>+9.1f}%")

        results[channel] = {
            'current_spend': current,
            'optimal_spend': optimal,
            'change_pct': change_pct,
            'roi': total_roi[channel]
        }

    return results


def plot_posterior_predictive(actual, pred_mean, pred_lower, pred_upper, dates):
    """Create posterior predictive check visualization."""
    fig, ax = plt.subplots(figsize=(14, 6))

    time_idx = np.arange(len(actual))

    # Plot actual sales
    ax.plot(time_idx, actual, 'o-', color='black', linewidth=2,
            markersize=3, label='Actual Sales', alpha=0.7)

    # Plot predicted mean
    ax.plot(time_idx, pred_mean, '-', color='#2E86AB', linewidth=2,
            label='Predicted Mean')

    # Plot credible interval
    ax.fill_between(time_idx, pred_lower, pred_upper,
                     color='#2E86AB', alpha=0.2,
                     label='95% Credible Interval')

    ax.set_xlabel('Week', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sales Revenue ($)', fontsize=12, fontweight='bold')
    ax.set_title('Posterior Predictive Check: Model Fit Quality',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_budget_optimization(results):
    """Create budget optimization visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    channels = list(results.keys())
    current_spend = [results[ch]['current_spend'] for ch in channels]
    optimal_spend = [results[ch]['optimal_spend'] for ch in channels]
    roi_values = [results[ch]['roi'] for ch in channels]

    # Plot 1: Current vs Optimal Allocation
    x = np.arange(len(channels))
    width = 0.35

    ax1.bar(x - width/2, current_spend, width, label='Current',
            color='#A23B72', alpha=0.7)
    ax1.bar(x + width/2, optimal_spend, width, label='Optimal',
            color='#18A558', alpha=0.7)

    ax1.set_xlabel('Marketing Channel', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Budget Allocation ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Budget Allocation: Current vs Optimal',
                  fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(channels, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: ROI vs Budget Change
    change_pct = [results[ch]['change_pct'] for ch in channels]
    colors = ['#18A558' if c > 0 else '#E63946' for c in change_pct]

    ax2.barh(channels, change_pct, color=colors, alpha=0.7)
    ax2.axvline(0, color='black', linewidth=1, linestyle='--')
    ax2.set_xlabel('Budget Change (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Recommended Budget Changes',
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    # Add ROI labels
    for i, (ch, roi) in enumerate(zip(channels, roi_values)):
        ax2.text(change_pct[i], i, f'  ROI: ${roi:.0f}',
                va='center', fontsize=9)

    plt.tight_layout()
    return fig


def main():
    print("="*80)
    print("ENHANCED MMM TEST - PRODUCTION CONFIGURATION")
    print("="*80)
    print("\nEnhancements:")
    print("  • 4 chains × 500 draws for robust convergence")
    print("  • Posterior predictive checks for model validation")
    print("  • Budget optimization for ROI maximization")
    print("  • Default PyTensor backend (JAX disabled for multiprocessing)")
    print()

    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("[1/10] Loading full prepared data (208 weeks)...")
    df = pd.read_csv('data/prepared_data.csv', parse_dates=['Date'])

    print(f"   ✓ Loaded {len(df)} weeks of data")
    print(f"   ✓ Date range: {df['Date'].min()} to {df['Date'].max()}")

    # Extract data components
    sales = df['revenue'].values
    dates = df['Date'].values

    marketing_channels = ['Content Marketing', 'Events', 'Google Ads', 'LinkedIn']
    marketing_data = df[marketing_channels].values

    brand_cols = ['Awareness', 'Consideration']
    brand_metrics = df[brand_cols].values

    # Control variables
    control_cols = ['Competitor_A_Spend', 'Competitor_B_Spend',
                    'GDP_Growth', 'Unemployment_Rate', 'Consumer_Confidence']
    control_data = df[control_cols].values

    print(f"   ✓ Marketing channels: {marketing_channels}")
    print(f"   ✓ Control variables: {control_cols}")

    # Calculate average spend per channel for budget optimization
    avg_spend = {ch: marketing_data[:, i].mean() for i, ch in enumerate(marketing_channels)}

    # =========================================================================
    # STEP 2: Build UCM-MMM Model
    # =========================================================================
    print("\n[2/10] Building OPTIMIZED UCM-MMM model...")

    channel_groups = {
        'digital': [0, 2, 3],  # Content Marketing, Google Ads, LinkedIn
        'offline': [1]          # Events
    }

    mmm = UCM_MMM_Optimized(
        sales_data=sales,
        marketing_data=marketing_data,
        control_data=control_data,
        marketing_channels=marketing_channels,
        control_names=control_cols,
        adstock_max_lag=8,
        channel_groups=channel_groups
    )

    mmm.build_model()
    print("   ✓ Model built successfully")
    print(f"   → Free parameters: {len(mmm.model.free_RVs)}")

    # =========================================================================
    # STEP 3: Fit UCM-MMM with Enhanced Configuration
    # =========================================================================
    print("\n[3/10] Fitting UCM-MMM with ENHANCED CONFIGURATION...")
    print("   → Configuration: 500 tuning + 500 draws × 4 chains")
    print("   → Target accept: 0.95 (very robust)")
    print("   → Estimated time: 15-25 minutes (without JAX)")
    print("   ⚠ This will take a while - patience is rewarded with better convergence!")

    start_time = time.time()

    mmm.fit(
        draws=500,
        tune=500,
        chains=4,
        target_accept=0.95
    )

    mmm_fit_time = time.time() - start_time
    print(f"   ✓ Model fitted in {mmm_fit_time:.1f}s ({mmm_fit_time/60:.1f} min)")

    # =========================================================================
    # STEP 4: Analyze UCM-MMM Results
    # =========================================================================
    print("\n[4/10] Analyzing UCM-MMM results...")

    summary = mmm.summary()
    rhat_max = summary['r_hat'].max()
    ess_min = summary['ess_bulk'].min()

    print(f"\n   --- Convergence Diagnostics ---")
    print(f"   Max R-hat: {rhat_max:.4f} (should be < 1.01)")
    print(f"   Min ESS:   {ess_min:.0f} (should be > 1000 for 500 draws × 4 chains)")

    if rhat_max < 1.01:
        print("   ✓ EXCELLENT convergence achieved!")
    elif rhat_max < 1.05:
        print("   ✓ Good convergence (acceptable for most uses)")
    else:
        print("   ⚠ Warning: Consider increasing draws further")

    # Calculate short-term ROI
    short_term_roi = mmm.calculate_short_term_roi()

    print("\n   --- Short-Term ROI (Immediate Activation) ---")
    for channel, roi in short_term_roi.items():
        print(f"   {channel:20s}: ${roi:.2f} per $1 spent")

    base_sales = mmm.get_base_sales()

    # =========================================================================
    # STEP 5: Posterior Predictive Check
    # =========================================================================
    print("\n[5/10] Performing Posterior Predictive Check...")

    ppc_results = posterior_predictive_check(mmm, sales, n_samples=200)

    # =========================================================================
    # STEP 6: Build and Fit BVAR Model
    # =========================================================================
    print("\n[6/10] Building OPTIMIZED BVAR model...")

    endog = np.column_stack([base_sales, brand_metrics])
    endog_names = ['Base_Sales'] + brand_cols

    bvar = BVAR_Optimized(
        endog=endog,
        exog=marketing_data,
        lags=2,
        endog_names=endog_names,
        exog_names=marketing_channels
    )

    bvar.build_model()
    print("   ✓ BVAR model built successfully")

    print("\n[7/10] Fitting BVAR with ENHANCED CONFIGURATION...")
    print("   → Configuration: 500 tuning + 500 draws × 4 chains")

    start_time = time.time()

    bvar.fit(
        draws=500,
        tune=500,
        chains=4,
        target_accept=0.95
    )

    bvar_fit_time = time.time() - start_time
    print(f"   ✓ BVAR fitted in {bvar_fit_time:.1f}s ({bvar_fit_time/60:.1f} min)")

    # =========================================================================
    # STEP 7: Calculate IRFs and Long-Term ROI
    # =========================================================================
    print("\n[8/10] Calculating Impulse Response Functions...")

    irf = bvar.calculate_irf(periods=24, shock_size=1.0, credible_interval=0.95)
    long_term_roi = bvar.calculate_long_term_roi(irf=irf, sales_var_name='Base_Sales')

    print("\n   --- Long-Term ROI (Brand-Building) with 95% CI ---")
    for channel, roi_dict in long_term_roi.items():
        mean_roi = roi_dict['mean']
        lower = roi_dict['lower']
        upper = roi_dict['upper']
        print(f"   {channel:20s}: ${mean_roi:>8,.2f}  [{lower:>8,.2f}, {upper:>9,.2f}]")

    # Calculate total ROI
    total_roi = {}
    for channel in marketing_channels:
        short_roi = short_term_roi.get(channel, 0)
        long_roi = long_term_roi.get(channel, {}).get('mean', 0)
        total_roi[channel] = short_roi + long_roi

    print("\n   --- TOTAL ROI (Short-Term + Long-Term) ---")
    for channel, roi in total_roi.items():
        print(f"   {channel:20s}: ${roi:>8,.2f} per $1 spent")

    # =========================================================================
    # STEP 8: Budget Optimization
    # =========================================================================
    print("\n[9/10] Optimizing Budget Allocation...")

    # Use total marketing budget (sum of average spends)
    total_budget = sum(avg_spend.values())

    optimization_results = optimize_budget(
        total_roi=total_roi,
        current_spend=avg_spend,
        total_budget=total_budget
    )

    # =========================================================================
    # STEP 9: Generate Visualizations
    # =========================================================================
    print("\n[10/10] Generating enhanced visualizations...")

    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)

    # Plot 1: Posterior predictive check
    print("   → Plotting posterior predictive check...")
    fig = plot_posterior_predictive(
        sales,
        ppc_results['y_pred_mean'],
        ppc_results['y_pred_lower'],
        ppc_results['y_pred_upper'],
        dates
    )
    fig.savefig(output_dir / 'posterior_predictive_check.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: outputs/posterior_predictive_check.png")
    plt.close(fig)

    # Plot 2: Budget optimization
    print("   → Plotting budget optimization...")
    fig = plot_budget_optimization(optimization_results)
    fig.savefig(output_dir / 'budget_optimization.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: outputs/budget_optimization.png")
    plt.close(fig)

    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "="*80)
    print("✓ ENHANCED TEST COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nPerformance:")
    print(f"  • Total time: {mmm_fit_time + bvar_fit_time:.1f}s ({(mmm_fit_time + bvar_fit_time)/60:.1f} min)")
    print(f"  • UCM-MMM: {mmm_fit_time:.1f}s")
    print(f"  • BVAR: {bvar_fit_time:.1f}s")

    print(f"\nModel Quality:")
    print(f"  • Max R-hat: {rhat_max:.4f}")
    print(f"  • MAPE: {ppc_results['mape']:.2f}%")
    print(f"  • R²: {ppc_results['r2']:.3f}")
    print(f"  • 95% CI Coverage: {ppc_results['coverage']:.1f}%")

    print(f"\nTop Channel by ROI:")
    best_channel = max(total_roi, key=total_roi.get)
    print(f"  • {best_channel}: ${total_roi[best_channel]:,.2f} per $1")

    print("\nGenerated files:")
    print("  • outputs/posterior_predictive_check.png")
    print("  • outputs/budget_optimization.png")


if __name__ == '__main__':
    main()
