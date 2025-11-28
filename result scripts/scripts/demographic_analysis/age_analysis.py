"""
Age Analysis Script
===================

Analyzes the relationship between participants' age and key performance metrics
using rigorous statistical methods.

Metrics analyzed:
- RAIR (Reliance on AI when AI is right)
- RSR (Resistance to wrong AI)
- Plausibility ratings
- Confidence change
- Final accuracy

Statistical Methods:
1. Spearman rank correlations (exploratory, participant-level aggregates)
2. Regression with cluster-robust standard errors (main analysis, trial-level data)
   - OLS for continuous outcomes (plausibility, confidence change)
   - Logistic regression for binary outcomes (accuracy, RAIR, RSR)
   - Clustering by participant accounts for repeated measures
   - Same methodology as h_tests.py

Visualizations:
- Scatter plots with age on x-axis
- Box plots by age groups
- Violin plots showing full distribution shapes
- AOR scatter plot (RAIR vs RSR) by age groups

Always uses unfiltered data (all participants regardless of job filter).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
import os
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))
from hypothesis_tests import make_long, ols_clustered, logit_clustered

# TUM Color Palette (from config)
TUM_BLUE = config.TUM_BLUE
TUM_BLUE_DARK = config.TUM_BLUE_DARK
TUM_BLUE_DARKER = config.TUM_BLUE_DARKER
TUM_ORANGE = config.TUM_ORANGE
TUM_GREEN = config.TUM_GREEN
TUM_LIGHT_BLUE = config.TUM_LIGHT_BLUE
TUM_MED_BLUE = config.TUM_MED_BLUE
TUM_BEIGE = config.TUM_BEIGE
TUM_GRAY_80 = config.TUM_GRAY_80
TUM_GRAY_50 = config.TUM_GRAY_50
TUM_GRAY_20 = config.TUM_GRAY_20

# Configuration
AGE_COLUMN = config.DEMOGRAPHIC_COLUMNS["age"]
DATA_FILE = config.PROCESSED_DATA_FILE
OUTPUT_FOLDER = config.PLOT_DIRS["age"]

# Age group mapping to ordinal numbers for regression
AGE_GROUP_MAPPING = config.AGE_GROUP_MAPPING


def add_age_to_long(long_df, df_wide, age_col):
    """
    Add participant-level age group to long format DataFrame.
    Age is stored as categorical groups, convert to ordinal for analysis.
    """
    # Create mapping from participant index to age group
    age_group_map = df_wide[age_col].to_dict()
    long_df['age_group'] = long_df['participant'].map(age_group_map)
    
    # Convert to ordinal for regression
    long_df['age_ordinal'] = long_df['age_group'].map(AGE_GROUP_MAPPING)
    return long_df


def compute_participant_level_metrics(long_df, df_wide):
    """
    Aggregate trial-level metrics to participant level and merge with wide format metrics.
    """
    # Aggregate from long format
    participant_agg = long_df.groupby('participant').agg({
        'plaus': 'mean',
        'delta_conf': 'mean',
        'age_group': 'first',
        'age_ordinal': 'first'
    }).reset_index()
    
    # Rename for clarity
    participant_agg.rename(columns={
        'plaus': 'Mean_Plausibility',
        'delta_conf': 'Mean_Conf_Delta'
    }, inplace=True)
    
    # Merge with wide format metrics
    wide_metrics = df_wide[['RAIR_user', 'RSR_user', 'Final_Accuracy_User']].copy()
    wide_metrics['participant'] = wide_metrics.index
    
    participant_metrics = participant_agg.merge(
        wide_metrics, 
        on='participant', 
        how='left'
    )
    
    return participant_metrics


def spearman_correlations(participant_df, age_col, metrics):
    """
    Compute Spearman rank correlations between age and all metrics.
    """
    results = []
    
    for metric in metrics:
        # Remove NaN values for this pair
        valid_data = participant_df[[age_col, metric]].dropna()
        
        if len(valid_data) < 3:
            results.append({
                'Metric': metric,
                'Spearman_rho': np.nan,
                'p_value': np.nan,
                'n': len(valid_data),
                'Significance': '',
                'Effect_Size': 'Insufficient data'
            })
            continue
        
        # Compute Spearman correlation
        rho, p_value = stats.spearmanr(valid_data[age_col], valid_data[metric])
        
        # Determine significance
        if p_value < 0.001:
            sig = '***'
        elif p_value < 0.01:
            sig = '**'
        elif p_value < 0.05:
            sig = '*'
        else:
            sig = ''
        
        # Interpret effect size
        abs_rho = abs(rho)
        if abs_rho < 0.1:
            effect = 'Negligible'
        elif abs_rho < 0.3:
            effect = 'Weak'
        elif abs_rho < 0.5:
            effect = 'Moderate'
        else:
            effect = 'Strong'
        
        results.append({
            'Metric': metric,
            'Spearman_rho': rho,
            'p_value': p_value,
            'n': len(valid_data),
            'Significance': sig,
            'Effect_Size': effect
        })
    
    return pd.DataFrame(results)


def plot_scatter_by_age(df, x_col, y_col, title, output_path):
    """
    Create scatter plot with age group (ordinal) on x-axis and trend line.
    """
    # Remove NaN values
    plot_data = df[[x_col, y_col]].dropna()
    
    if len(plot_data) == 0:
        print(f"Warning: No valid data for {title}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot
    ax.scatter(plot_data[x_col], plot_data[y_col], alpha=0.5, s=60, color=TUM_BLUE, edgecolors=TUM_BLUE_DARK, linewidth=0.5)
    
    # Add regression line
    if len(plot_data) >= 3:
        z = np.polyfit(plot_data[x_col], plot_data[y_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(plot_data[x_col].min(), plot_data[x_col].max(), 100)
        ax.plot(x_line, p(x_line), color=TUM_ORANGE, linewidth=2.5, linestyle='-', label='Trend line')
    
    # Compute correlation for annotation
    rho, p_value = stats.spearmanr(plot_data[x_col], plot_data[y_col])
    sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
    
    # Add correlation annotation
    ax.text(0.98, 0.02, f'Spearman ρ = {rho:.3f} {sig}\np = {p_value:.4f}\nn = {len(plot_data)}',
            transform=ax.transAxes, fontsize=11, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=TUM_GRAY_50))
    
    ax.set_xlabel('Age Group (1=18-24, 2=25-34, 3=35-44, 4=45-54, 5=55+)', fontsize=11, fontweight='bold')
    ax.set_ylabel(y_col.replace('_', ' '), fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.2, linestyle='--')
    if len(plot_data) >= 3:
        ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_boxplot_by_age_group(df, y_col, title, output_path):
    """
    Create box plot with age groups on x-axis.
    """
    # Remove NaN values
    plot_data = df[['age_group', y_col]].dropna()
    
    if len(plot_data) == 0:
        print(f"Warning: No valid data for {title}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define age group order
    age_order = ["18-24", "25-34", "35-44", "45-54", "55+"]
    available_groups = [g for g in age_order if g in plot_data['age_group'].unique()]
    
    # Create box plot
    box_data = [plot_data[plot_data['age_group'] == group][y_col].values for group in available_groups]
    positions = list(range(len(available_groups)))
    
    bp = ax.boxplot(box_data, positions=positions, widths=0.5, patch_artist=True,
                     boxprops=dict(facecolor=TUM_LIGHT_BLUE, alpha=0.7, edgecolor=TUM_BLUE_DARK),
                     whiskerprops=dict(color=TUM_BLUE_DARK, linewidth=1.5),
                     capprops=dict(color=TUM_BLUE_DARK, linewidth=1.5),
                     medianprops=dict(color=TUM_ORANGE, linewidth=2.5),
                     flierprops=dict(marker='o', markerfacecolor=TUM_GRAY_50, markersize=6, alpha=0.5))
    
    # Overlay individual points with jitter
    for i, group in enumerate(available_groups):
        group_data = plot_data[plot_data['age_group'] == group][y_col].values
        jitter = np.random.normal(0, 0.08, len(group_data))
        ax.scatter([i]*len(group_data) + jitter, group_data, alpha=0.4, s=40, color=TUM_BLUE, edgecolors='none')
    
    # Add mean line
    means = [plot_data[plot_data['age_group'] == group][y_col].mean() for group in available_groups]
    ax.plot(positions, means, color=TUM_GREEN, linewidth=2.5, marker='D', markersize=8, 
            label='Mean', linestyle='--', zorder=10)
    
    # Add sample size annotations
    for i, group in enumerate(available_groups):
        n = len(plot_data[plot_data['age_group'] == group])
        ax.text(i, ax.get_ylim()[1] * 0.98, f'n={n}', 
                ha='center', va='top', fontsize=9, color=TUM_GRAY_80)
    
    ax.set_xlabel('Age Group', fontsize=13, fontweight='bold')
    ax.set_ylabel(y_col.replace('_', ' '), fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(positions)
    ax.set_xticklabels(available_groups)
    ax.grid(True, alpha=0.2, linestyle='--', axis='y')
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_metrics_by_age(participant_df, output_path):
    """
    Create separate subplots for each metric showing distributions by age group (scatter plots).
    """
    metrics = ['RAIR_user', 'RSR_user', 'Mean_Plausibility', 'Mean_Conf_Delta', 'Final_Accuracy_User']
    metric_labels = ['RAIR', 'RSR', 'Mean Plausibility', 'Mean Confidence Change', 'Final Accuracy']
    colors = [TUM_BLUE, TUM_ORANGE, TUM_GREEN, TUM_MED_BLUE, TUM_BEIGE]
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        ax = axes[idx]
        plot_data = participant_df[['age_ordinal', metric]].dropna()
        
        if len(plot_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Scatter plot
        ax.scatter(plot_data['age_ordinal'], plot_data[metric], alpha=0.5, s=50, color=color, edgecolors='black', linewidth=0.5)
        
        # Add trend line
        if len(plot_data) >= 3:
            z = np.polyfit(plot_data['age_ordinal'], plot_data[metric], 1)
            p = np.poly1d(z)
            x_line = np.linspace(plot_data['age_ordinal'].min(), plot_data['age_ordinal'].max(), 100)
            ax.plot(x_line, p(x_line), color='red', linewidth=2, linestyle='--', alpha=0.7, label='Trend')
        
        # Compute correlation
        rho, p_value = stats.spearmanr(plot_data['age_ordinal'], plot_data[metric])
        sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        
        # Add stats annotation
        ax.text(0.02, 0.98, f'ρ = {rho:.3f} {sig}', transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Age Group', fontsize=11, fontweight='bold')
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle='--')
    
    # Remove the 6th subplot
    fig.delaxes(axes[5])
    
    plt.suptitle('Performance Metrics by Age Group', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_aor_scatter_by_age_group(long_df, output_path):
    """
    Plot AOR scatter: RAIR (x-axis) vs RSR (y-axis) with one point per age group.
    AOR (Appropriateness of Reliance) = (RAIR + RSR) / 2
    """
    # age_group already exists in long_df
    
    # RAIR-eligible: AI correct & human initially wrong
    df_rair = long_df[(long_df["ai_correct"]==1) & (long_df["human_pre_correct"]==0)].copy()
    df_rair["rair"] = df_rair["changed_to_correct"].astype(float)
    rair_by_age = df_rair.groupby("age_group")["rair"].mean().rename("RAIR")
    
    # RSR-eligible: AI wrong & human initially correct
    df_rsr = long_df[(long_df["ai_correct"]==0) & (long_df["human_pre_correct"]==1)].copy()
    df_rsr["rsr"] = df_rsr["stayed_correct"].astype(float)
    rsr_by_age = df_rsr.groupby("age_group")["rsr"].mean().rename("RSR")
    
    # Combine
    summary = pd.concat([rair_by_age, rsr_by_age], axis=1).reset_index()
    summary = summary.dropna()  # Remove any age groups with missing data
    summary["AOR"] = (summary["RAIR"] + summary["RSR"]) / 2.0
    
    if len(summary) == 0:
        print("Warning: No valid data for AOR plot")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Define age group order and colors
    age_order = ["18-24", "25-34", "35-44", "45-54", "55+"]
    colors = [TUM_BLUE_DARKER, TUM_BLUE_DARK, TUM_BLUE, TUM_MED_BLUE, TUM_LIGHT_BLUE]
    
    # Filter to available groups and maintain order
    summary['age_group'] = pd.Categorical(summary['age_group'], categories=age_order, ordered=True)
    summary = summary.sort_values('age_group').dropna(subset=['age_group'])  # Remove NaN age groups
    
    color_map = {group: colors[age_order.index(group)] for group in summary['age_group'] if group in age_order}
    point_colors = [color_map[group] for group in summary['age_group']]
    
    # Scatter plot
    scatter = ax.scatter(summary["RAIR"], summary["RSR"], s=300, c=point_colors, 
                        edgecolor="black", linewidth=2, zorder=10, alpha=0.8)
    
    # Annotate each point with age group and AOR
    for _, row in summary.iterrows():
        label = f"{row['age_group']}\nAOR={row['AOR']:.3f}"
        
        # Position labels to avoid overlap
        xytext = (10, 10)
        
        ax.annotate(label, (row["RAIR"], row["RSR"]), 
                   xytext=xytext, textcoords="offset points",
                   ha="left", va="bottom",
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="white", 
                            edgecolor=color_map[row["age_group"]], alpha=0.95, linewidth=2))
    
    # Add diagonal line for reference
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1.5, label='RAIR = RSR')
    
    # Formatting
    ax.set_xlabel('RAIR (Reliance on AI when Right)', fontsize=13, fontweight='bold')
    ax.set_ylabel('RSR (Resistance to wrong AI)', fontsize=13, fontweight='bold')
    ax.set_title('Appropriateness of Reliance (AOR) by Age Group\nRAIR vs RSR', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    
    # Add text box with AOR explanation
    explanation = "AOR = (RAIR + RSR) / 2\nHigher values = Better reliance"
    ax.text(0.02, 0.98, explanation, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor=TUM_BEIGE, alpha=0.8))
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', 
                             label=group,
                             markerfacecolor=color_map[group], 
                             markeredgecolor='black', markersize=10)
                      for group in summary['age_group']]
    ax.legend(handles=legend_elements, loc='lower right', title='Age Group', 
             framealpha=0.95, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\n" + "="*80)
    print("AOR (Appropriateness of Reliance) by Age Group")
    print("="*80)
    print(f"{'Age Group':<15} {'RAIR':<10} {'RSR':<10} {'AOR':<10}")
    print("-"*80)
    for _, row in summary.iterrows():
        print(f"{row['age_group']:<15} {row['RAIR']:<10.3f} {row['RSR']:<10.3f} {row['AOR']:<10.3f}")
    print("="*80)
    
    print(f"\nSaved AOR plot to: {output_path}")


def regression_analysis_clustered(long_df, age_col='age'):
    """
    Perform regression analyses with cluster-robust standard errors.
    Uses the same methodology as h_tests.py to account for repeated measures.
    """
    results = {}
    
    # Prepare data - ensure age is numeric
    analysis_df = long_df.copy()
    analysis_df[age_col] = pd.to_numeric(analysis_df[age_col], errors='coerce')
    
    print("\n" + "="*80)
    print("REGRESSION ANALYSES WITH CLUSTER-ROBUST STANDARD ERRORS")
    print("="*80)
    print("Clustering by participant to account for repeated measures")
    print("Same methodology as h_tests.py")
    
    # 1. Plausibility (continuous) - OLS with cluster-robust SEs
    print("\n" + "-"*80)
    print("1. PLAUSIBILITY ~ Age (OLS with cluster-robust SEs)")
    print("-"*80)
    try:
        model_plaus = ols_clustered(f'plaus ~ {age_col}', data=analysis_df, cluster_var='participant')
        results['plausibility'] = model_plaus
        print(model_plaus.summary())
        coef = model_plaus.params[age_col]
        pval = model_plaus.pvalues[age_col]
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        print(f"\nCoefficient: {coef:.4f} {sig}")
        print(f"Interpretation: Each 1-level increase in age group is associated with")
        print(f"               {abs(coef):.4f} {'increase' if coef > 0 else 'decrease'} in plausibility rating")
    except Exception as e:
        print(f"Error: {e}")
        results['plausibility'] = None
    
    # 2. Confidence change (continuous) - OLS with cluster-robust SEs
    print("\n" + "-"*80)
    print("2. CONFIDENCE_CHANGE ~ Age (OLS with cluster-robust SEs)")
    print("-"*80)
    try:
        model_conf = ols_clustered(f'delta_conf ~ {age_col}', data=analysis_df, cluster_var='participant')
        results['confidence_change'] = model_conf
        print(model_conf.summary())
        coef = model_conf.params[age_col]
        pval = model_conf.pvalues[age_col]
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        print(f"\nCoefficient: {coef:.4f} {sig}")
        print(f"Interpretation: Each 1-level increase in age group is associated with")
        print(f"               {abs(coef):.4f} {'increase' if coef > 0 else 'decrease'} in confidence change")
    except Exception as e:
        print(f"Error: {e}")
        results['confidence_change'] = None
    
    # 3. Human final accuracy (binary) - Logistic regression with cluster-robust SEs
    print("\n" + "-"*80)
    print("3. FINAL_ACCURACY (post==gt) ~ Age (Logistic with cluster-robust SEs)")
    print("-"*80)
    try:
        analysis_df['final_correct'] = (analysis_df['post'] == analysis_df['gt']).astype(int)
        model_acc = logit_clustered(f'final_correct ~ {age_col}', data=analysis_df, cluster_var='participant')
        results['final_accuracy'] = model_acc
        print(model_acc.summary())
        coef = model_acc.params[age_col]
        pval = model_acc.pvalues[age_col]
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        odds_ratio = np.exp(coef)
        print(f"\nLog-odds coefficient: {coef:.4f} {sig}")
        print(f"Odds ratio: {odds_ratio:.4f}")
        print(f"Interpretation: Each 1-level increase in age group multiplies the odds")
        print(f"               of being correct by {odds_ratio:.4f}")
    except Exception as e:
        print(f"Error: {e}")
        results['final_accuracy'] = None
    
    # 4. RAIR subset - changed_to_correct among AI-correct & human initially wrong
    print("\n" + "-"*80)
    print("4. RAIR (changed_to_correct | AI correct & human initially wrong)")
    print("   Logistic regression with cluster-robust SEs")
    print("-"*80)
    try:
        rair_df = analysis_df[(analysis_df['ai_correct']==1) & (analysis_df['human_pre_correct']==0)].copy()
        print(f"RAIR-eligible trials: {len(rair_df)}")
        if len(rair_df) >= 10:
            model_rair = logit_clustered(f'changed_to_correct ~ {age_col}', data=rair_df, cluster_var='participant')
            results['rair'] = model_rair
            print(model_rair.summary())
            coef = model_rair.params[age_col]
            pval = model_rair.pvalues[age_col]
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            odds_ratio = np.exp(coef)
            print(f"\nLog-odds coefficient: {coef:.4f} {sig}")
            print(f"Odds ratio: {odds_ratio:.4f}")
            print(f"Interpretation: Each 1-level increase in age group multiplies the odds")
            print(f"               of accepting AI's correct suggestion by {odds_ratio:.4f}")
        else:
            print("Insufficient RAIR-eligible observations for regression")
            results['rair'] = None
    except Exception as e:
        print(f"Error: {e}")
        results['rair'] = None
    
    # 5. RSR subset - stayed_correct among AI-wrong & human initially correct
    print("\n" + "-"*80)
    print("5. RSR (stayed_correct | AI wrong & human initially correct)")
    print("   Logistic regression with cluster-robust SEs")
    print("-"*80)
    try:
        rsr_df = analysis_df[(analysis_df['ai_correct']==0) & (analysis_df['human_pre_correct']==1)].copy()
        print(f"RSR-eligible trials: {len(rsr_df)}")
        if len(rsr_df) >= 10:
            model_rsr = logit_clustered(f'stayed_correct ~ {age_col}', data=rsr_df, cluster_var='participant')
            results['rsr'] = model_rsr
            print(model_rsr.summary())
            coef = model_rsr.params[age_col]
            pval = model_rsr.pvalues[age_col]
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            odds_ratio = np.exp(coef)
            print(f"\nLog-odds coefficient: {coef:.4f} {sig}")
            print(f"Odds ratio: {odds_ratio:.4f}")
            print(f"Interpretation: Each 1-level increase in age group multiplies the odds")
            print(f"               of resisting AI's wrong suggestion by {odds_ratio:.4f}")
        else:
            print("Insufficient RSR-eligible observations for regression")
            results['rsr'] = None
    except Exception as e:
        print(f"Error: {e}")
        results['rsr'] = None
    
    print("\n" + "="*80)
    
    return results


def save_correlation_summary(results_df, output_path):
    """Save correlation results to CSV file."""
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved correlation summary to: {output_path}")


def print_correlation_summary(results_df):
    """Print formatted correlation summary to console."""
    print("\n" + "="*80)
    print("AGE CORRELATION ANALYSIS")
    print("="*80)
    print("\nSpearman Rank Correlations (Non-parametric)")
    print("Significance: * p<.05, ** p<.01, *** p<.001")
    print("Effect Size: |ρ| < 0.1 (Negligible), 0.1-0.3 (Weak), 0.3-0.5 (Moderate), > 0.5 (Strong)")
    print("\n" + "-"*80)
    
    for _, row in results_df.iterrows():
        print(f"\n{row['Metric']}:")
        print(f"  Spearman ρ = {row['Spearman_rho']:.4f} {row['Significance']}")
        print(f"  p-value = {row['p_value']:.4f}")
        print(f"  n = {row['n']}")
        print(f"  Effect size: {row['Effect_Size']}")
    
    print("\n" + "="*80)


def main():
    """Main analysis function."""
    print("="*80)
    print("AGE ANALYSIS")
    print("="*80)
    print(f"\nLoading data from: {DATA_FILE}")
    print("Note: Always uses unfiltered data (all participants)\n")
    
    # Create output folder if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output folder: {OUTPUT_FOLDER}/")
    else:
        print(f"Using output folder: {OUTPUT_FOLDER}/")
    
    # Load data
    try:
        df_wide = pd.read_excel(str(DATA_FILE))
        print(f"Loaded {len(df_wide)} participants")
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found. Please run script.py first.")
        return
    
    # Check if age column exists
    if AGE_COLUMN not in df_wide.columns:
        print(f"\nError: Column '{AGE_COLUMN}' not found in data.")
        print("Available columns containing 'age':")
        relevant_cols = [c for c in df_wide.columns if 'age' in c.lower()]
        for col in relevant_cols:
            print(f"  - {col}")
        return
    
    # Create long format
    print("\nConverting to long format...")
    long_df = make_long(df_wide, n_trials=config.N_TRIALS)
    print(f"Created {len(long_df)} trial observations")
    
    # Add age to long format
    long_df = add_age_to_long(long_df, df_wide, AGE_COLUMN)
    
    # Compute participant-level metrics
    print("\nComputing participant-level metrics...")
    participant_metrics = compute_participant_level_metrics(long_df, df_wide)
    print(f"Aggregated data for {len(participant_metrics)} participants")
    
    # Check age distribution
    age_group_dist = participant_metrics['age_group'].value_counts().sort_index()
    print("\nAge Group Distribution:")
    for group, count in age_group_dist.items():
        print(f"  {group}: {count} participants")
    
    # Define metrics to analyze
    metrics = ['RAIR_user', 'RSR_user', 'Mean_Plausibility', 'Mean_Conf_Delta', 'Final_Accuracy_User']
    
    # Perform Spearman correlations (exploratory analysis)
    print("\n" + "="*80)
    print("PART 1: SPEARMAN CORRELATIONS (Exploratory)")
    print("="*80)
    print("Performing Spearman rank correlations on participant-level aggregates...")
    print("Using age_ordinal (1-5) for correlation analysis")
    correlation_results = spearman_correlations(participant_metrics, 'age_ordinal', metrics)
    
    # Print summary
    print_correlation_summary(correlation_results)
    
    # Save correlation summary
    csv_path = os.path.join(OUTPUT_FOLDER, 'age_correlations.csv')
    save_correlation_summary(correlation_results, csv_path)
    
    # Perform regression analyses with cluster-robust SEs (main analysis)
    print("\n" + "="*80)
    print("PART 2: REGRESSION ANALYSES (Main Analysis)")
    print("="*80)
    print("Using trial-level data with cluster-robust SEs (same as h_tests.py)")
    print("Age groups treated as ordinal (1=18-24, 2=25-34, 3=35-44, 4=45-54, 5=55+)")
    regression_results = regression_analysis_clustered(long_df, age_col='age_ordinal')
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Scatter plots (age ordinal as continuous)
    scatter_configs = [
        ('RAIR_user', 'Reliance on AI when AI is Right (RAIR) by Age Group', 'age_vs_rair_scatter.png'),
        ('RSR_user', 'Resistance to Wrong AI (RSR) by Age Group', 'age_vs_rsr_scatter.png'),
        ('Mean_Plausibility', 'Mean Plausibility Rating by Age Group', 'age_vs_plausibility_scatter.png'),
        ('Mean_Conf_Delta', 'Mean Confidence Change by Age Group', 'age_vs_conf_delta_scatter.png'),
        ('Final_Accuracy_User', 'Final Accuracy by Age Group', 'age_vs_accuracy_scatter.png')
    ]
    
    for metric, title, filename in scatter_configs:
        plot_path = os.path.join(OUTPUT_FOLDER, filename)
        plot_scatter_by_age(participant_metrics, 'age_ordinal', metric, title, plot_path)
    
    # Box plots by age groups
    boxplot_configs = [
        ('RAIR_user', 'RAIR by Age Group', 'age_group_vs_rair.png'),
        ('RSR_user', 'RSR by Age Group', 'age_group_vs_rsr.png'),
        ('Mean_Plausibility', 'Mean Plausibility by Age Group', 'age_group_vs_plausibility.png'),
        ('Mean_Conf_Delta', 'Mean Confidence Change by Age Group', 'age_group_vs_conf_delta.png'),
        ('Final_Accuracy_User', 'Final Accuracy by Age Group', 'age_group_vs_accuracy.png')
    ]
    
    for metric, title, filename in boxplot_configs:
        plot_path = os.path.join(OUTPUT_FOLDER, filename)
        plot_boxplot_by_age_group(participant_metrics, metric, title, plot_path)
    
    # Multi-panel plot
    grouped_plot_path = os.path.join(OUTPUT_FOLDER, 'age_metrics_scatter.png')
    plot_metrics_by_age(participant_metrics, grouped_plot_path)
    
    # AOR scatter plot
    print("\nCreating AOR scatter plot...")
    aor_plot_path = os.path.join(OUTPUT_FOLDER, 'age_aor_scatter.png')
    plot_aor_scatter_by_age_group(long_df, aor_plot_path)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll files saved in folder: {OUTPUT_FOLDER}/")
    print("\nStatistical Analyses Performed:")
    print("  1. Spearman correlations (exploratory, participant-level)")
    print("  2. Regression with cluster-robust SEs (main analysis, trial-level)")
    print("     - Accounts for repeated measures within participants")
    print("     - Same methodology as h_tests.py")
    print("\nVisualization Types:")
    print("  - Scatter plots (age as continuous variable)")
    print("  - Box plots by age groups")
    print("  - Multi-panel scatter plots")
    print("  - AOR scatter plot by age groups")
    print("\nGenerated files:")
    print(f"  - {OUTPUT_FOLDER}/age_correlations.csv (correlation summary)")
    print(f"  - {OUTPUT_FOLDER}/age_vs_*_scatter.png (5 scatter plots)")
    print(f"  - {OUTPUT_FOLDER}/age_group_vs_*.png (5 box plots by age group)")
    print(f"  - {OUTPUT_FOLDER}/age_metrics_scatter.png (multi-panel scatter plots)")
    print(f"  - {OUTPUT_FOLDER}/age_aor_scatter.png (AOR scatter: RAIR vs RSR)")
    print("\nRegression results are printed above (full statsmodels output)")
    print("\n")


if __name__ == "__main__":
    main()
