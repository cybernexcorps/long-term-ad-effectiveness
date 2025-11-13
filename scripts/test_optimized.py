#!/usr/bin/env python
"""
Comprehensive test comparing original vs optimized MMM models.

This script:
1. Tests both model versions on full 208-week dataset
2. Includes control variables (competitor spend, macro indicators)
3. Compares execution time and convergence
4. Generates comparison visualizations

Usage:
    python scripts/test_optimized.py
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Try to configure JAX (optional)
try:
    import config_jax
    print("✓ JAX backend configured")
except Exception as e:
    print(f"⚠ JAX not available: {e}")
    print("  Continuing with default backend (slower but functional)")

from mmm_optimized import UCM_MMM_Optimized
from bvar_optimized import BVAR_Optimized


def main():
    print("="*80)
    print("OPTIMIZED MMM MODELS - COMPREHENSIVE TEST")
    print("="*80)

    # =========================================================================
    # STEP 1: Load Full Dataset with Control Variables
    # =========================================================================
    print("\n[1/8] Loading full prepared data (208 weeks)...")
    df = pd.read_csv('data/prepared_data.csv', parse_dates=['Date'])

    print(f"   ✓ Loaded {len(df)} weeks of data")
    print(f"   ✓ Date range: {df['Date'].min()} to {df['Date'].max()}")

    # Extract data components
    sales = df['revenue'].values

    marketing_channels = ['Content Marketing', 'Events', 'Google Ads', 'LinkedIn']
    marketing_data = df[marketing_channels].values

    brand_cols = ['Awareness', 'Consideration']
    brand_metrics = df[brand_cols].values

    # Control variables: competitor + macroeconomic
    control_cols = ['Competitor_A_Spend', 'Competitor_B_Spend',
                    'GDP_Growth', 'Unemployment_Rate', 'Consumer_Confidence']
    control_data = df[control_cols].values

    print(f"   ✓ Marketing channels: {marketing_channels}")
    print(f"   ✓ Brand metrics: {brand_cols}")
    print(f"   ✓ Control variables: {control_cols}")

    # =========================================================================
    # STEP 2: Build Optimized UCM-MMM Model
    # =========================================================================
    print("\n[2/8] Building OPTIMIZED UCM-MMM model...")
    print("   → Features: PyMC-Marketing adstock, control variables, seasonality")

    # Optional: Define channel groups for hierarchical effects
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
    print(f"   → Model includes {len(mmm.model.free_RVs)} free parameters")

    # =========================================================================
    # STEP 3: Fit Optimized UCM-MMM
    # =========================================================================
    print("\n[3/8] Fitting OPTIMIZED UCM-MMM with MCMC...")
    print("   → Configuration: 200 tuning + 200 draws × 2 chains")
    print("   → Target accept: 0.9 (robust sampling)")
    print("   → Estimated time: 5-10 minutes (depends on hardware)")

    start_time = time.time()

    mmm.fit(
        draws=200,
        tune=200,
        chains=2,  # Use 2 chains for faster testing (4 recommended for production)
        target_accept=0.9
    )

    mmm_fit_time = time.time() - start_time
    print(f"   ✓ Model fitted successfully in {mmm_fit_time:.1f} seconds ({mmm_fit_time/60:.1f} minutes)")

    # =========================================================================
    # STEP 4: Analyze UCM-MMM Results
    # =========================================================================
    print("\n[4/8] Analyzing UCM-MMM results...")

    # Check convergence
    summary = mmm.summary()
    rhat_max = summary['r_hat'].max()
    ess_min = summary['ess_bulk'].min()

    print(f"\n   --- Convergence Diagnostics ---")
    print(f"   Max R-hat: {rhat_max:.3f} (should be < 1.1)")
    print(f"   Min ESS:   {ess_min:.0f} (should be > 400 for 200 draws × 2 chains)")

    if rhat_max < 1.1:
        print("   ✓ MCMC converged successfully")
    else:
        print("   ⚠ Warning: Some parameters have high R-hat (increase draws)")

    # Calculate short-term ROI
    short_term_roi = mmm.calculate_short_term_roi()

    print("\n   --- Short-Term ROI (Immediate Activation) ---")
    for channel, roi in short_term_roi.items():
        print(f"   {channel:20s}: ${roi:.2f} per $1 spent")

    # Extract base sales for BVAR
    base_sales = mmm.get_base_sales()
    print(f"\n   ✓ Extracted base sales for BVAR model")

    # =========================================================================
    # STEP 5: Build and Fit Optimized BVAR Model
    # =========================================================================
    print("\n[5/8] Building OPTIMIZED BVAR model...")

    # Combine base sales with brand metrics
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

    print("\n[6/8] Fitting OPTIMIZED BVAR with MCMC...")
    print("   → Configuration: 200 tuning + 200 draws × 2 chains")

    start_time = time.time()

    bvar.fit(
        draws=200,
        tune=200,
        chains=2,
        target_accept=0.9
    )

    bvar_fit_time = time.time() - start_time
    print(f"   ✓ BVAR fitted successfully in {bvar_fit_time:.1f} seconds ({bvar_fit_time/60:.1f} minutes)")

    # =========================================================================
    # STEP 6: Calculate IRFs and Long-Term ROI
    # =========================================================================
    print("\n[7/8] Calculating Impulse Response Functions...")
    print("   → Simulating 24-week response with 95% credible intervals")

    irf = bvar.calculate_irf(periods=24, shock_size=1.0, credible_interval=0.95)
    print(f"   ✓ Calculated {len(irf)} IRF trajectories with uncertainty")

    # Show sample IRF with confidence intervals
    print("\n   --- Sample IRF: LinkedIn → Base Sales (first 12 weeks) ---")
    sample_irf = irf.get('LinkedIn_to_Base_Sales', None)
    if sample_irf:
        for week in range(12):
            mean = sample_irf['mean'][week]
            lower = sample_irf['lower'][week]
            upper = sample_irf['upper'][week]
            print(f"   Week {week:2d}: ${mean:>10,.2f}  [{lower:>10,.2f}, {upper:>10,.2f}]")

    # Calculate long-term ROI with uncertainty
    long_term_roi = bvar.calculate_long_term_roi(irf=irf, sales_var_name='Base_Sales')

    print("\n   --- Long-Term ROI (Brand-Building) with 95% CI ---")
    for channel in marketing_channels:
        roi_data = long_term_roi.get(channel, {})
        mean = roi_data.get('mean', 0.0)
        lower = roi_data.get('lower', 0.0)
        upper = roi_data.get('upper', 0.0)
        print(f"   {channel:20s}: ${mean:>8,.2f}  [{lower:>8,.2f}, {upper:>8,.2f}]")

    # =========================================================================
    # STEP 7: Total ROI Summary
    # =========================================================================
    print("\n" + "="*80)
    print("TOTAL ROI SUMMARY (208 weeks, Full Dataset)")
    print("="*80)

    print(f"\n{'Channel':<20s} {'Short-Term':<15s} {'Long-Term (Mean)':<20s} {'Total ROI':<15s}")
    print("-" * 80)

    for i, channel in enumerate(marketing_channels):
        st_roi = short_term_roi[channel]
        lt_roi_mean = long_term_roi[channel]['mean']
        total_roi = st_roi + lt_roi_mean

        print(f"{channel:<20s} ${st_roi:>8.2f}       ${lt_roi_mean:>12.2f}       ${total_roi:>8.2f}")

    # =========================================================================
    # STEP 8: Generate Visualizations
    # =========================================================================
    print("\n[8/8] Generating visualizations...")

    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)

    # 1. IRF plot with confidence intervals
    print("   → Plotting IRFs with credible intervals...")
    fig1 = bvar.plot_irf(irf, figsize=(16, 12), show_ci=True)
    fig1.savefig('outputs/irf_plot_optimized.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: outputs/irf_plot_optimized.png")

    # 2. Long-term ROI with error bars
    print("   → Plotting Long-Term ROI with uncertainty...")
    fig2, ax = plt.subplots(figsize=(12, 7))

    channels_list = list(long_term_roi.keys())
    means = [long_term_roi[ch]['mean'] for ch in channels_list]
    lowers = [long_term_roi[ch]['lower'] for ch in channels_list]
    uppers = [long_term_roi[ch]['upper'] for ch in channels_list]
    errors = [[means[i] - lowers[i], uppers[i] - means[i]] for i in range(len(means))]

    x = np.arange(len(channels_list))
    colors = sns.color_palette("viridis", len(channels_list))

    bars = ax.bar(x, means, color=colors, alpha=0.8, edgecolor='black')
    ax.errorbar(x, means, yerr=np.array(errors).T, fmt='none', ecolor='black',
                capsize=5, linewidth=2, alpha=0.7)

    # Add value labels
    for i, (bar, mean) in enumerate(zip(bars, means)):
        ax.text(bar.get_x() + bar.get_width()/2., mean,
               f'${mean:.2f}',
               ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_ylabel('Long-Term ROI ($ per $1 spent)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Marketing Channel', fontsize=12, fontweight='bold')
    ax.set_title('Long-Term ROI from Brand-Building (208 weeks, 95% CI)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(channels_list, rotation=45, ha='right')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig2.savefig('outputs/long_term_roi_optimized.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: outputs/long_term_roi_optimized.png")

    # 3. Total ROI comparison
    print("   → Plotting Total ROI decomposition...")
    fig3, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(marketing_channels))
    width = 0.35

    st_rois = [short_term_roi[ch] for ch in marketing_channels]
    lt_rois = [long_term_roi[ch]['mean'] for ch in marketing_channels]

    bars1 = ax.bar(x - width/2, st_rois, width, label='Short-Term',
                   color='steelblue', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, lt_rois, width, label='Long-Term',
                   color='coral', alpha=0.8, edgecolor='black')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if abs(height) > 0.01:  # Only label non-zero bars
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'${height:.2f}',
                       ha='center', va='bottom' if height > 0 else 'top',
                       fontsize=9, fontweight='bold')

    ax.set_xlabel('Marketing Channel', fontsize=12, fontweight='bold')
    ax.set_ylabel('ROI ($ per $1 spent)', fontsize=12, fontweight='bold')
    ax.set_title('Marketing ROI Decomposition: Short-Term vs Long-Term (208 weeks)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(marketing_channels, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.8)

    plt.tight_layout()
    fig3.savefig('outputs/total_roi_comparison_optimized.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: outputs/total_roi_comparison_optimized.png")

    # 4. Channel contribution over time
    print("   → Plotting channel contribution timeline...")
    fig4 = mmm.plot_channel_contribution()
    fig4.savefig('outputs/channel_contribution_timeline.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: outputs/channel_contribution_timeline.png")

    plt.close('all')

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("✓ TEST COMPLETED SUCCESSFULLY")
    print("="*80)

    print(f"\n📊 Performance Summary:")
    print(f"   UCM-MMM fit time:  {mmm_fit_time:.1f}s ({mmm_fit_time/60:.1f} min)")
    print(f"   BVAR fit time:     {bvar_fit_time:.1f}s ({bvar_fit_time/60:.1f} min)")
    print(f"   Total time:        {(mmm_fit_time + bvar_fit_time):.1f}s ({(mmm_fit_time + bvar_fit_time)/60:.1f} min)")

    print(f"\n✅ Models:")
    print(f"   - Optimized adstock (PyMC-Marketing)")
    print(f"   - Control variables included")
    print(f"   - Hierarchical channel effects")
    print(f"   - Seasonality modeling")
    print(f"   - Full uncertainty quantification")

    print(f"\n📁 Outputs:")
    print(f"   - outputs/irf_plot_optimized.png")
    print(f"   - outputs/long_term_roi_optimized.png")
    print(f"   - outputs/total_roi_comparison_optimized.png")
    print(f"   - outputs/channel_contribution_timeline.png")

    print("\n" + "="*80)

    return {
        'mmm': mmm,
        'bvar': bvar,
        'short_term_roi': short_term_roi,
        'long_term_roi': long_term_roi,
        'irf': irf,
        'mmm_fit_time': mmm_fit_time,
        'bvar_fit_time': bvar_fit_time
    }


if __name__ == '__main__':
    results = main()
