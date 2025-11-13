#!/usr/bin/env python
"""
Test script for MMM and BVAR models on synthetic data.

This script demonstrates the complete workflow:
1. Load prepared data
2. Fit UCM-MMM short-term model
3. Extract base sales
4. Fit BVAR long-term model
5. Calculate IRFs and long-term ROI
6. Compute total ROI (short-term + long-term)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mmm import UCM_MMM
from bvar import BVAR

def main():
    print("="*80)
    print("MARKETING MIX MODELING - COMPLETE PIPELINE TEST")
    print("="*80)

    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("\n[1/7] Loading prepared data...")
    df = pd.read_csv('data/prepared_data.csv', parse_dates=['Date'])
    print(f"   ✓ Loaded {len(df)} weeks of data")
    print(f"   ✓ Date range: {df['Date'].min()} to {df['Date'].max()}")

    # Extract components
    sales = df['revenue'].values
    marketing_channels = ['Content Marketing', 'Events', 'Google Ads', 'LinkedIn']
    marketing_data = df[marketing_channels].values

    brand_cols = ['Awareness', 'Consideration']
    brand_metrics = df[brand_cols].values

    print(f"   ✓ Marketing channels: {marketing_channels}")
    print(f"   ✓ Brand metrics: {brand_cols}")

    # =========================================================================
    # STEP 2: Build and Fit UCM-MMM (Short-Term Model)
    # =========================================================================
    print("\n[2/7] Building UCM-MMM model...")
    print("   → Model: Sales = Baseline + Trend + Adstock(Marketing) + Saturation")

    mmm = UCM_MMM(
        sales_data=sales,
        marketing_data=marketing_data,
        max_lag=4
    )
    mmm.build_model()
    print("   ✓ Model built successfully")

    print("\n[3/7] Fitting UCM-MMM with MCMC sampling...")
    print("   → Running 500 tuning + 500 sampling iterations...")
    print("   (This may take a few minutes)")

    mmm.fit(draws=500, tune=500)
    print("   ✓ Model fitted successfully")

    # =========================================================================
    # STEP 3: Extract Short-Term Results
    # =========================================================================
    print("\n[4/7] Calculating short-term ROI...")

    # Get summary statistics
    summary = mmm.summary()
    print("\n   --- UCM-MMM Posterior Summary ---")
    print(summary[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat']].head(15))

    # Calculate short-term ROI
    short_term_roi = mmm.calculate_short_term_roi()
    print("\n   --- Short-Term ROI (Immediate Activation) ---")
    for i, channel in enumerate(marketing_channels):
        roi_value = short_term_roi[f'channel_{i}']
        print(f"   {channel:20s}: ${roi_value:.2f} per $1 spent")

    # Extract base sales for BVAR
    base_sales = mmm.get_base_sales()
    print(f"\n   ✓ Extracted base sales (trend component)")

    # =========================================================================
    # STEP 4: Build and Fit BVAR (Long-Term Model)
    # =========================================================================
    print("\n[5/7] Building BVAR model...")
    print("   → Model: [Base Sales, Awareness, Consideration] = VAR + Marketing")

    # Combine base sales with brand metrics for endogenous variables
    endog = np.column_stack([base_sales, brand_metrics])
    endog_names = ['Base_Sales'] + brand_cols

    bvar = BVAR(
        endog=endog,
        exog=marketing_data,
        lags=2,
        endog_names=endog_names,
        exog_names=marketing_channels
    )
    bvar.build_model()
    print("   ✓ BVAR model built successfully")

    print("\n   Fitting BVAR with MCMC sampling...")
    print("   → Running 500 tuning + 500 sampling iterations...")

    bvar.fit(draws=500, tune=500)
    print("   ✓ BVAR model fitted successfully")

    # =========================================================================
    # STEP 5: Calculate Impulse Response Functions
    # =========================================================================
    print("\n[6/7] Calculating Impulse Response Functions...")
    print("   → Simulating 24-week response to $1 marketing shock")

    irf = bvar.calculate_irf(periods=24, shock_size=1.0)
    print(f"   ✓ Calculated {len(irf)} IRF trajectories")

    # Show sample IRF values
    print("\n   --- Sample IRF: LinkedIn → Base Sales (first 12 weeks) ---")
    sample_irf = irf.get('LinkedIn_to_Base_Sales', None)
    if sample_irf is not None:
        for week in range(12):
            print(f"   Week {week:2d}: ${sample_irf[week]:,.2f}")

    # =========================================================================
    # STEP 6: Calculate Long-Term ROI
    # =========================================================================
    print("\n[7/7] Calculating long-term ROI...")

    long_term_roi = bvar.calculate_long_term_roi(
        irf=irf,
        sales_var_name='Base_Sales'
    )

    print("\n   --- Long-Term ROI (Brand-Building Effects) ---")
    for channel in marketing_channels:
        roi_value = long_term_roi.get(channel, 0.0)
        print(f"   {channel:20s}: ${roi_value:.2f} per $1 spent")

    # =========================================================================
    # STEP 7: Calculate Total ROI
    # =========================================================================
    print("\n" + "="*80)
    print("TOTAL ROI SUMMARY (Short-Term + Long-Term)")
    print("="*80)

    print(f"\n{'Channel':<20s} {'Short-Term':<15s} {'Long-Term':<15s} {'Total ROI':<15s}")
    print("-" * 70)

    for i, channel in enumerate(marketing_channels):
        st_roi = short_term_roi[f'channel_{i}']
        lt_roi = long_term_roi.get(channel, 0.0)
        total_roi = st_roi + lt_roi

        print(f"{channel:<20s} ${st_roi:>8.2f}       ${lt_roi:>8.2f}       ${total_roi:>8.2f}")

    print("\n" + "="*80)
    print("✓ PIPELINE TEST COMPLETED SUCCESSFULLY")
    print("="*80)

    # =========================================================================
    # STEP 8: Create Visualizations
    # =========================================================================
    print("\n[BONUS] Creating visualizations...")

    # Plot IRFs
    print("   → Plotting Impulse Response Functions...")
    fig1 = bvar.plot_irf(irf, figsize=(16, 12))
    fig1.savefig('outputs/irf_plot.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: outputs/irf_plot.png")

    # Plot long-term ROI
    print("   → Plotting Long-Term ROI...")
    fig2 = bvar.plot_long_term_roi(long_term_roi)
    fig2.savefig('outputs/long_term_roi.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: outputs/long_term_roi.png")

    # Create total ROI comparison plot
    print("   → Plotting Total ROI Comparison...")
    fig3, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(marketing_channels))
    width = 0.35

    st_rois = [short_term_roi[f'channel_{i}'] for i in range(len(marketing_channels))]
    lt_rois = [long_term_roi.get(ch, 0.0) for ch in marketing_channels]

    bars1 = ax.bar(x - width/2, st_rois, width, label='Short-Term', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, lt_rois, width, label='Long-Term', color='coral', alpha=0.8)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${height:.2f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Marketing Channel', fontsize=12, fontweight='bold')
    ax.set_ylabel('ROI ($ per $1 spent)', fontsize=12, fontweight='bold')
    ax.set_title('Marketing ROI Decomposition: Short-Term vs Long-Term', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(marketing_channels, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.8)

    plt.tight_layout()
    fig3.savefig('outputs/total_roi_comparison.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: outputs/total_roi_comparison.png")

    print("\n✓ All visualizations saved to outputs/ directory")

    # Close plots to free memory
    plt.close('all')

    return {
        'mmm': mmm,
        'bvar': bvar,
        'short_term_roi': short_term_roi,
        'long_term_roi': long_term_roi,
        'irf': irf
    }

if __name__ == '__main__':
    results = main()
