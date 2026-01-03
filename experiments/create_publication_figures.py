"""
Create publication-quality figures for paper
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

def create_main_results_figure():
    """Figure 1: Main results across all datasets"""
    
    # Load all results
    results = [
        {'dataset': 'NASA\nTurbofan', 'drift': 'Linear (0.5)', 'std': 50.8, 'adp': 89.3},
        {'dataset': 'NASA\nTurbofan', 'drift': 'Stepwise (0.8)', 'std': 47.2, 'adp': 88.4},
        {'dataset': 'Bearing', 'drift': 'Replacement', 'std': 63.7, 'adp': 82.3},
        {'dataset': 'Wind\nTurbine', 'drift': 'Documented', 'std': 67.5, 'adp': 88.7},
        {'dataset': 'Gas\nSensors', 'drift': 'Real 3-year', 'std': 57.5, 'adp': 66.0},
    ]
    
    df = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df['std'], width, label='Standard Conformal', 
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, df['adp'], width, label='Adaptive Conformal', 
                   color='#27ae60', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Target line
    ax.axhline(90, color='black', linestyle='--', linewidth=2, label='Target (90%)', zorder=0)
    
    # Add improvement annotations
    for i, (std, adp) in enumerate(zip(df['std'], df['adp'])):
        improvement = adp - std
        ax.annotate(f'+{improvement:.1f}pp', 
                   xy=(i, max(std, adp) + 2),
                   ha='center', fontsize=11, fontweight='bold',
                   color='green' if improvement > 10 else 'orange')
    
    ax.set_ylabel('Coverage (%)', fontsize=14, fontweight='bold')
    ax.set_title('Adaptive Conformal Maintains Coverage Under Drift', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['dataset']}\n({row['drift']})" for _, row in df.iterrows()],
                       fontsize=10)
    ax.legend(fontsize=12, loc='lower right')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('figures/fig1_main_results.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig1_main_results.pdf', bbox_inches='tight')
    print("✓ Figure 1: Main results")

def create_drift_severity_figure():
    """Figure 2: Performance vs drift severity"""
    
    drift_levels = [0.0, 0.2, 0.3, 0.5, 0.75, 1.0]
    std_coverage = [91.0, 86.5, 80.8, 50.8, 49.8, 20.1]
    adp_coverage = [89.7, 89.4, 89.4, 89.3, 89.0, 88.4]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(drift_levels, std_coverage, 'o-', linewidth=3, markersize=10,
            label='Standard Conformal', color='#e74c3c', markeredgecolor='black', markeredgewidth=1.5)
    ax.plot(drift_levels, adp_coverage, 's-', linewidth=3, markersize=10,
            label='Adaptive Conformal', color='#27ae60', markeredgecolor='black', markeredgewidth=1.5)
    
    ax.axhline(90, color='black', linestyle='--', linewidth=2, label='Target', alpha=0.7)
    
    ax.set_xlabel('Drift Magnitude', fontsize=14, fontweight='bold')
    ax.set_ylabel('Coverage (%)', fontsize=14, fontweight='bold')
    ax.set_title('Adaptive Conformal is Robust to Drift Severity', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    # Highlight catastrophic degradation region
    ax.fill_between([0.5, 1.0], 0, 100, alpha=0.1, color='red', 
                    label='_Catastrophic degradation\n(Standard Conformal)')
    
    plt.tight_layout()
    plt.savefig('figures/fig2_drift_severity.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig2_drift_severity.pdf', bbox_inches='tight')
    print("✓ Figure 2: Drift severity")

def create_stability_figure():
    """Figure 3: Stability comparison (variance)"""
    
    # Wind turbine 5-seed results
    seeds = list(range(1, 6))
    std_vals = [58.9, 59.6, 70.7, 89.9, 58.2]
    adp_vals = [88.6, 91.2, 88.8, 87.2, 87.9]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Individual seeds
    ax1.plot(seeds, std_vals, 'o-', linewidth=2.5, markersize=12,
            label='Standard Conformal', color='#e74c3c', markeredgecolor='black', markeredgewidth=1.5)
    ax1.plot(seeds, adp_vals, 's-', linewidth=2.5, markersize=12,
            label='Adaptive Conformal', color='#27ae60', markeredgecolor='black', markeredgewidth=1.5)
    
    ax1.axhline(90, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Random Seed', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Coverage (%)', fontsize=13, fontweight='bold')
    ax1.set_title('(a) Coverage Across Random Seeds', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([50, 95])
    
    # Right: Variance comparison
    methods = ['Standard\nConformal', 'Adaptive\nConformal']
    means = [np.mean(std_vals), np.mean(adp_vals)]
    stds = [np.std(std_vals), np.std(adp_vals)]
    
    x_pos = np.arange(len(methods))
    colors = ['#e74c3c', '#27ae60']
    
    bars = ax2.bar(x_pos, means, yerr=stds, capsize=10, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5,
                   error_kw={'linewidth': 2.5, 'ecolor': 'black'})
    
    ax2.axhline(90, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_ylabel('Coverage (%)', fontsize=13, fontweight='bold')
    ax2.set_title('(b) Mean ± Std Dev', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods, fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([50, 95])
    
    # Add variance labels
    ax2.text(0, means[0] + stds[0] + 3, f'σ = {stds[0]:.1f}%', 
            ha='center', fontsize=11, fontweight='bold', color='darkred')
    ax2.text(1, means[1] + stds[1] + 3, f'σ = {stds[1]:.1f}%', 
            ha='center', fontsize=11, fontweight='bold', color='darkgreen')
    
    plt.tight_layout()
    plt.savefig('figures/fig3_stability.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig3_stability.pdf', bbox_inches='tight')
    print("✓ Figure 3: Stability analysis")

def create_baseline_comparison_figure():
    """Figure 4: Baseline method comparison"""
    
    methods = ['Standard\nConformal', 'Online\nLearning', 'Sliding\nWindow', 'Adaptive\nConformal\n(Ours)']
    coverage = [50.8, 62.9, 88.7, 89.4]
    colors = ['#e74c3c', '#f39c12', '#3498db', '#27ae60']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(methods, coverage, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=1.5)
    
    ax.axhline(90, color='black', linestyle='--', linewidth=2, label='Target')
    
    # Add value labels
    for bar, val in zip(bars, coverage):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Coverage (%)', fontsize=14, fontweight='bold')
    ax.set_title('Comparison with Baseline Methods (NASA C-MAPSS, drift=0.5)', 
                fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig('figures/fig4_baselines.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig4_baselines.pdf', bbox_inches='tight')
    print("✓ Figure 4: Baseline comparison")

def create_cost_benefit_figure():
    """Figure 5: Economic analysis"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Annual cost breakdown
    categories = ['Standard\nApproach\n(12x/year)', 'Adaptive\nApproach\n(4x/year)']
    calibration = [253200, 84400]
    computational = [0, 4380]
    
    x = np.arange(len(categories))
    width = 0.6
    
    bars1 = ax1.bar(x, calibration, width, label='Calibration Cost', 
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x, computational, width, bottom=calibration,
                   label='Computational Cost', color='#3498db', alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('Annual Cost ($)', fontsize=13, fontweight='bold')
    ax1.set_title('(a) Annual Cost per Sensor', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add savings annotation
    savings = 253200 - 88780
    ax1.annotate(f'Savings:\n${savings:,}/year', 
                xy=(0.5, 170000), fontsize=13, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7, edgecolor='black', linewidth=2),
                fontweight='bold')
    
    # Right: Sensitivity to downtime cost
    downtime_costs = [1000, 5000, 10000, 20000]
    savings_vals = [36420, 164420, 324420, 644420]
    
    ax2.plot(downtime_costs, np.array(savings_vals)/1000, 'o-', 
            linewidth=3, markersize=12, color='#27ae60',
            markeredgecolor='black', markeredgewidth=1.5)
    
    ax2.set_xlabel('Production Downtime Cost ($/hour)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Annual Savings ($1000s)', fontsize=13, fontweight='bold')
    ax2.set_title('(b) Sensitivity to Downtime Cost', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/fig5_economics.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/fig5_economics.pdf', bbox_inches='tight')
    print("✓ Figure 5: Economic analysis")

def main():
    import os
    os.makedirs('figures', exist_ok=True)
    
    print("="*70)
    print("Creating Publication-Quality Figures")
    print("="*70)
    
    create_main_results_figure()
    create_drift_severity_figure()
    create_stability_figure()
    create_baseline_comparison_figure()
    create_cost_benefit_figure()
    
    print(f"\n{'='*70}")
    print("✓ All figures created in figures/")
    print("  - PNG (300 DPI) for viewing")
    print("  - PDF (vector) for publication")
    print("="*70)

if __name__ == '__main__':
    main()
