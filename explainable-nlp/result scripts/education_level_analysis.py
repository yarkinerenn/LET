"""
Education Level Analysis Script
================================

Analyzes the relationship between participants' education level and key performance metrics
using rigorous statistical methods.

Education Levels (ordinal):
- High school diploma (1)
- Bachelor's degree (2)
- Master's degree (3)
- PhD or equivalent (4)

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
- Box plots with overlaid individual data points
- Violin plots showing full distribution shapes
- Mean and median lines clearly marked
- Sample sizes for each education level
- AOR scatter plot (RAIR vs RSR)

Always uses unfiltered data (all participants regardless of job filter).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from h_tests import make_long, ols_clustered, logit_clustered
import statsmodels.formula.api as smf
import os

# TUM Color Palette (from plots.py)
TUM_BLUE = "#0065BD"
TUM_BLUE_DARK = "#005293"
TUM_BLUE_DARKER = "#003359"
TUM_ORANGE = "#E37222"
TUM_GREEN = "#A2AD00"
TUM_LIGHT_BLUE = "#98C6EA"
TUM_MED_BLUE = "#64A0C8"
TUM_BEIGE = "#DAD7CB"
TUM_GRAY_80 = "#333333"
TUM_GRAY_50 = "#808080"
TUM_GRAY_20 = "#CCCCCC"

# Configuration
EDU_COLUMN = "What is your highest achieved level of education?"
DATA_FILE = 'experiment_results_with_metrics.xlsx'
OUTPUT_FOLDER = 'education_level_plots'

# Map education levels to ordinal numbers
EDU_MAPPING = {
    "High school diploma": 1,
    "Bachelor's degree": 2,
    "Master's degree": 3,
    "PhD or equivalent": 4,
    "Phd or equivalent": 4  # Handle possible case variation
}

# Reverse mapping for labels
EDU_LABELS = {
    1: "High School",
    2: "Bachelor's",
    3: "Master's",
    4: "PhD"
}


def add_education_to_long(long_df, df_wide, edu_col):
    """
    Add participant-level education level to long format DataFrame.
    Convert categorical education to ordinal numbers.
    """
    # Create mapping from participant index to education
    edu_map = df_wide[edu_col].map(EDU_MAPPING).to_dict()
    long_df['education_level'] = long_df['participant'].map(edu_map)
    return long_df


def compute_participant_level_metrics(long_df, df_wide):
    """
    Aggregate trial-level metrics to participant level and merge with wide format metrics.
    """
    # Aggregate from long format
    participant_agg = long_df.groupby('participant').agg({
        'plaus': 'mean',
        'delta_conf': 'mean',
        'education_level': 'first'
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


def spearman_correlations(participant_df, edu_col, metrics):
    """
    Compute Spearman rank correlations between education level and all metrics.
    """
    results = []
    
    for metric in metrics:
        # Remove NaN values for this pair
        valid_data = participant_df[[edu_col, metric]].dropna()
        
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
        rho, p_value = stats.spearmanr(valid_data[edu_col], valid_data[metric])
        
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


def plot_boxplot_by_education(df, x_col, y_col, title, output_path):
    """
    Create box plot with overlaid points showing distribution by education level.
    """
    # Remove NaN values
    plot_data = df[[x_col, y_col]].dropna()
    
    if len(plot_data) == 0:
        print(f"Warning: No valid data for {title}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create box plot
    positions = sorted(plot_data[x_col].unique())
    box_data = [plot_data[plot_data[x_col] == pos][y_col].values for pos in positions]
    
    bp = ax.boxplot(box_data, positions=positions, widths=0.4, patch_artist=True,
                     boxprops=dict(facecolor=TUM_LIGHT_BLUE, alpha=0.7, edgecolor=TUM_BLUE_DARK),
                     whiskerprops=dict(color=TUM_BLUE_DARK, linewidth=1.5),
                     capprops=dict(color=TUM_BLUE_DARK, linewidth=1.5),
                     medianprops=dict(color=TUM_ORANGE, linewidth=2.5),
                     flierprops=dict(marker='o', markerfacecolor=TUM_GRAY_50, markersize=6, alpha=0.5))
    
    # Overlay individual points with jitter
    for pos in positions:
        pos_data = plot_data[plot_data[x_col] == pos][y_col].values
        jitter = np.random.normal(0, 0.06, len(pos_data))
        ax.scatter(pos + jitter, pos_data, alpha=0.4, s=40, color=TUM_BLUE, edgecolors='none')
    
    # Add mean line
    means = [plot_data[plot_data[x_col] == pos][y_col].mean() for pos in positions]
    ax.plot(positions, means, color=TUM_GREEN, linewidth=2.5, marker='D', markersize=8, 
            label='Mean', linestyle='--', zorder=10)
    
    # Add sample size annotations
    for pos in positions:
        n = len(plot_data[plot_data[x_col] == pos])
        ax.text(pos, ax.get_ylim()[1] * 0.98, f'n={n}', 
                ha='center', va='top', fontsize=9, color=TUM_GRAY_80)
    
    # Compute correlation for annotation
    rho, p_value = stats.spearmanr(plot_data[x_col], plot_data[y_col])
    sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
    
    # Add correlation annotation
    ax.text(0.98, 0.02, f'Spearman ρ = {rho:.3f} {sig}\np = {p_value:.4f}\nTotal n = {len(plot_data)}',
            transform=ax.transAxes, fontsize=11, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=TUM_GRAY_50))
    
    ax.set_xlabel('Education Level', fontsize=13, fontweight='bold')
    ax.set_ylabel(y_col.replace('_', ' '), fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim(0.5, 4.5)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels([EDU_LABELS[i] for i in [1, 2, 3, 4]])
    ax.grid(True, alpha=0.2, linestyle='--', axis='y')
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_metrics_by_education(participant_df, output_path):
    """
    Create separate subplots for each metric showing clear distributions by education level.
    """
    metrics = ['RAIR_user', 'RSR_user', 'Mean_Plausibility', 'Mean_Conf_Delta', 'Final_Accuracy_User']
    metric_labels = ['RAIR', 'RSR', 'Mean Plausibility', 'Mean Confidence Change', 'Final Accuracy']
    colors = [TUM_BLUE, TUM_ORANGE, TUM_GREEN, TUM_MED_BLUE, TUM_BEIGE]
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        ax = axes[idx]
        plot_data = participant_df[['education_level', metric]].dropna()
        
        if len(plot_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Create violin plot
        positions = sorted(plot_data['education_level'].unique())
        violin_data = [plot_data[plot_data['education_level'] == pos][metric].values for pos in positions]
        
        parts = ax.violinplot(violin_data, positions=positions, widths=0.5, 
                              showmeans=True, showmedians=True, showextrema=True)
        
        # Style violin plots
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
            pc.set_edgecolor(TUM_BLUE_DARK)
        
        parts['cmeans'].set_color(TUM_GREEN)
        parts['cmeans'].set_linewidth(2)
        parts['cmedians'].set_color(TUM_ORANGE)
        parts['cmedians'].set_linewidth(2)
        
        # Overlay individual points
        for pos in positions:
            pos_data = plot_data[plot_data['education_level'] == pos][metric].values
            jitter = np.random.normal(0, 0.05, len(pos_data))
            ax.scatter(pos + jitter, pos_data, alpha=0.3, s=25, color='black', edgecolors='none')
        
        # Add trend line
        z = np.polyfit(plot_data['education_level'], plot_data[metric], 1)
        p = np.poly1d(z)
        x_line = np.linspace(1, 4, 100)
        ax.plot(x_line, p(x_line), color='red', linewidth=2, linestyle='--', alpha=0.7, label='Trend')
        
        # Compute correlation
        rho, p_value = stats.spearmanr(plot_data['education_level'], plot_data[metric])
        sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        
        # Add stats annotation
        ax.text(0.02, 0.98, f'ρ = {rho:.3f} {sig}', transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Education Level', fontsize=11, fontweight='bold')
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_xlim(0.5, 4.5)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels([EDU_LABELS[i] for i in [1, 2, 3, 4]], fontsize=9)
        ax.grid(True, alpha=0.2, linestyle='--', axis='y')
    
    # Remove the 6th subplot
    fig.delaxes(axes[5])
    
    plt.suptitle('Performance Metrics by Education Level', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_aor_scatter_by_education(long_df, output_path):
    """
    Plot AOR scatter: RAIR (x-axis) vs RSR (y-axis) with one point per education level.
    AOR (Appropriateness of Reliance) = (RAIR + RSR) / 2
    """
    # RAIR-eligible: AI correct & human initially wrong
    df_rair = long_df[(long_df["ai_correct"]==1) & (long_df["human_pre_correct"]==0)].copy()
    df_rair["rair"] = df_rair["changed_to_correct"].astype(float)
    rair_by_edu = df_rair.groupby("education_level")["rair"].mean().rename("RAIR")
    
    # RSR-eligible: AI wrong & human initially correct
    df_rsr = long_df[(long_df["ai_correct"]==0) & (long_df["human_pre_correct"]==1)].copy()
    df_rsr["rsr"] = df_rsr["stayed_correct"].astype(float)
    rsr_by_edu = df_rsr.groupby("education_level")["rsr"].mean().rename("RSR")
    
    # Combine
    summary = pd.concat([rair_by_edu, rsr_by_edu], axis=1).reset_index()
    summary = summary.dropna()  # Remove any education levels with missing data
    summary["AOR"] = (summary["RAIR"] + summary["RSR"]) / 2.0
    summary["edu_label"] = summary["education_level"].map(EDU_LABELS)
    
    if len(summary) == 0:
        print("Warning: No valid data for AOR plot")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Color gradient for education levels
    colors = [TUM_BLUE_DARKER, TUM_BLUE_DARK, TUM_MED_BLUE, TUM_LIGHT_BLUE]
    edu_levels = sorted(summary["education_level"].unique())
    color_map = {level: colors[int(level)-1] for level in edu_levels}
    point_colors = [color_map[edu] for edu in summary["education_level"]]
    
    # Scatter plot
    scatter = ax.scatter(summary["RAIR"], summary["RSR"], s=300, c=point_colors, 
                        edgecolor="black", linewidth=2, zorder=10, alpha=0.8)
    
    # Annotate each point with level and AOR
    for _, row in summary.iterrows():
        level = int(row["education_level"])
        label = f"{row['edu_label']}\nAOR={row['AOR']:.3f}"
        
        # Position labels to avoid overlap
        if level == 1:
            xytext = (10, -15)
            va = 'top'
            ha = 'left'
        elif level == 2:
            xytext = (-15, -15)
            va = 'top'
            ha = 'right'
        elif level == 3:
            xytext = (10, 10)
            va = 'bottom'
            ha = 'left'
        else:  # PhD
            xytext = (-15, 10)
            va = 'bottom'
            ha = 'right'
        
        ax.annotate(label, (row["RAIR"], row["RSR"]), 
                   xytext=xytext, textcoords="offset points",
                   ha=ha, va=va,
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="white", 
                            edgecolor=color_map[row["education_level"]], alpha=0.95, linewidth=2))
    
    # Add diagonal line for reference
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1.5, label='RAIR = RSR')
    
    # Formatting
    ax.set_xlabel('RAIR (Reliance on AI when Right)', fontsize=13, fontweight='bold')
    ax.set_ylabel('RSR (Resistance to wrong AI)', fontsize=13, fontweight='bold')
    ax.set_title('Appropriateness of Reliance (AOR) by Education Level\nRAIR vs RSR', 
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
                             label=EDU_LABELS[int(level)],
                             markerfacecolor=color_map[level], 
                             markeredgecolor='black', markersize=10)
                      for level in edu_levels]
    ax.legend(handles=legend_elements, loc='lower right', title='Education Level', 
             framealpha=0.95, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\n" + "="*80)
    print("AOR (Appropriateness of Reliance) by Education Level")
    print("="*80)
    print(f"{'Level':<15} {'RAIR':<10} {'RSR':<10} {'AOR':<10}")
    print("-"*80)
    for _, row in summary.iterrows():
        print(f"{row['edu_label']:<15} {row['RAIR']:<10.3f} {row['RSR']:<10.3f} {row['AOR']:<10.3f}")
    print("="*80)
    
    print(f"\nSaved AOR plot to: {output_path}")


def regression_analysis_clustered(long_df, edu_col='education_level'):
    """
    Perform regression analyses with cluster-robust standard errors.
    Uses the same methodology as h_tests.py to account for repeated measures.
    """
    results = {}
    
    # Prepare data - ensure education_level is numeric
    analysis_df = long_df.copy()
    analysis_df[edu_col] = pd.to_numeric(analysis_df[edu_col], errors='coerce')
    
    print("\n" + "="*80)
    print("REGRESSION ANALYSES WITH CLUSTER-ROBUST STANDARD ERRORS")
    print("="*80)
    print("Clustering by participant to account for repeated measures")
    print("Same methodology as h_tests.py")
    
    # 1. Plausibility (continuous) - OLS with cluster-robust SEs
    print("\n" + "-"*80)
    print("1. PLAUSIBILITY ~ Education_Level (OLS with cluster-robust SEs)")
    print("-"*80)
    try:
        model_plaus = ols_clustered(f'plaus ~ {edu_col}', data=analysis_df, cluster_var='participant')
        results['plausibility'] = model_plaus
        print(model_plaus.summary())
        coef = model_plaus.params[edu_col]
        pval = model_plaus.pvalues[edu_col]
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        print(f"\nCoefficient: {coef:.4f} {sig}")
        print(f"Interpretation: Each 1-level increase in education is associated with")
        print(f"               {abs(coef):.4f} {'increase' if coef > 0 else 'decrease'} in plausibility rating")
    except Exception as e:
        print(f"Error: {e}")
        results['plausibility'] = None
    
    # 2. Confidence change (continuous) - OLS with cluster-robust SEs
    print("\n" + "-"*80)
    print("2. CONFIDENCE_CHANGE ~ Education_Level (OLS with cluster-robust SEs)")
    print("-"*80)
    try:
        model_conf = ols_clustered(f'delta_conf ~ {edu_col}', data=analysis_df, cluster_var='participant')
        results['confidence_change'] = model_conf
        print(model_conf.summary())
        coef = model_conf.params[edu_col]
        pval = model_conf.pvalues[edu_col]
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        print(f"\nCoefficient: {coef:.4f} {sig}")
        print(f"Interpretation: Each 1-level increase in education is associated with")
        print(f"               {abs(coef):.4f} {'increase' if coef > 0 else 'decrease'} in confidence change")
    except Exception as e:
        print(f"Error: {e}")
        results['confidence_change'] = None
    
    # 3. Human final accuracy (binary) - Logistic regression with cluster-robust SEs
    print("\n" + "-"*80)
    print("3. FINAL_ACCURACY (post==gt) ~ Education_Level (Logistic with cluster-robust SEs)")
    print("-"*80)
    try:
        analysis_df['final_correct'] = (analysis_df['post'] == analysis_df['gt']).astype(int)
        model_acc = logit_clustered(f'final_correct ~ {edu_col}', data=analysis_df, cluster_var='participant')
        results['final_accuracy'] = model_acc
        print(model_acc.summary())
        coef = model_acc.params[edu_col]
        pval = model_acc.pvalues[edu_col]
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        odds_ratio = np.exp(coef)
        print(f"\nLog-odds coefficient: {coef:.4f} {sig}")
        print(f"Odds ratio: {odds_ratio:.4f}")
        print(f"Interpretation: Each 1-level increase in education multiplies the odds")
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
            model_rair = logit_clustered(f'changed_to_correct ~ {edu_col}', data=rair_df, cluster_var='participant')
            results['rair'] = model_rair
            print(model_rair.summary())
            coef = model_rair.params[edu_col]
            pval = model_rair.pvalues[edu_col]
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            odds_ratio = np.exp(coef)
            print(f"\nLog-odds coefficient: {coef:.4f} {sig}")
            print(f"Odds ratio: {odds_ratio:.4f}")
            print(f"Interpretation: Each 1-level increase in education multiplies the odds")
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
            model_rsr = logit_clustered(f'stayed_correct ~ {edu_col}', data=rsr_df, cluster_var='participant')
            results['rsr'] = model_rsr
            print(model_rsr.summary())
            coef = model_rsr.params[edu_col]
            pval = model_rsr.pvalues[edu_col]
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            odds_ratio = np.exp(coef)
            print(f"\nLog-odds coefficient: {coef:.4f} {sig}")
            print(f"Odds ratio: {odds_ratio:.4f}")
            print(f"Interpretation: Each 1-level increase in education multiplies the odds")
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
    print("EDUCATION LEVEL CORRELATION ANALYSIS")
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
    print("EDUCATION LEVEL ANALYSIS")
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
        df_wide = pd.read_excel(DATA_FILE)
        print(f"Loaded {len(df_wide)} participants")
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found. Please run script.py first.")
        return
    
    # Check if education column exists
    if EDU_COLUMN not in df_wide.columns:
        print(f"\nError: Column '{EDU_COLUMN}' not found in data.")
        print("Available columns containing 'education':")
        relevant_cols = [c for c in df_wide.columns if 'education' in c.lower()]
        for col in relevant_cols:
            print(f"  - {col}")
        return
    
    # Create long format
    print("\nConverting to long format...")
    long_df = make_long(df_wide, n_trials=16)
    print(f"Created {len(long_df)} trial observations")
    
    # Add education level to long format
    long_df = add_education_to_long(long_df, df_wide, EDU_COLUMN)
    
    # Compute participant-level metrics
    print("\nComputing participant-level metrics...")
    participant_metrics = compute_participant_level_metrics(long_df, df_wide)
    print(f"Aggregated data for {len(participant_metrics)} participants")
    
    # Check education level distribution
    edu_dist = participant_metrics['education_level'].value_counts().sort_index()
    print("\nEducation Level Distribution:")
    for level, count in edu_dist.items():
        if not np.isnan(level):
            print(f"  {EDU_LABELS.get(int(level), 'Unknown')}: {count} participants")
    
    # Define metrics to analyze
    metrics = ['RAIR_user', 'RSR_user', 'Mean_Plausibility', 'Mean_Conf_Delta', 'Final_Accuracy_User']
    
    # Perform Spearman correlations (exploratory analysis)
    print("\n" + "="*80)
    print("PART 1: SPEARMAN CORRELATIONS (Exploratory)")
    print("="*80)
    print("Performing Spearman rank correlations on participant-level aggregates...")
    correlation_results = spearman_correlations(participant_metrics, 'education_level', metrics)
    
    # Print summary
    print_correlation_summary(correlation_results)
    
    # Save correlation summary
    csv_path = os.path.join(OUTPUT_FOLDER, 'education_correlations.csv')
    save_correlation_summary(correlation_results, csv_path)
    
    # Perform regression analyses with cluster-robust SEs (main analysis)
    print("\n" + "="*80)
    print("PART 2: REGRESSION ANALYSES (Main Analysis)")
    print("="*80)
    print("Using trial-level data with cluster-robust SEs (same as h_tests.py)")
    regression_results = regression_analysis_clustered(long_df, edu_col='education_level')
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Individual box plots
    plot_configs = [
        ('RAIR_user', 'Reliance on AI when AI is Right (RAIR) by Education Level', 'edu_vs_rair.png'),
        ('RSR_user', 'Resistance to Wrong AI (RSR) by Education Level', 'edu_vs_rsr.png'),
        ('Mean_Plausibility', 'Mean Plausibility Rating by Education Level', 'edu_vs_plausibility.png'),
        ('Mean_Conf_Delta', 'Mean Confidence Change by Education Level', 'edu_vs_conf_delta.png'),
        ('Final_Accuracy_User', 'Final Accuracy by Education Level', 'edu_vs_accuracy.png')
    ]
    
    for metric, title, filename in plot_configs:
        plot_path = os.path.join(OUTPUT_FOLDER, filename)
        plot_boxplot_by_education(participant_metrics, 'education_level', metric, title, plot_path)
    
    # Multi-panel plot
    grouped_plot_path = os.path.join(OUTPUT_FOLDER, 'edu_metrics_by_level.png')
    plot_metrics_by_education(participant_metrics, grouped_plot_path)
    
    # AOR scatter plot
    print("\nCreating AOR scatter plot...")
    aor_plot_path = os.path.join(OUTPUT_FOLDER, 'edu_aor_scatter.png')
    plot_aor_scatter_by_education(long_df, aor_plot_path)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll files saved in folder: {OUTPUT_FOLDER}/")
    print("\nStatistical Analyses Performed:")
    print("  1. Spearman correlations (exploratory, participant-level)")
    print("  2. Regression with cluster-robust SEs (main analysis, trial-level)")
    print("     - Accounts for repeated measures within participants")
    print("     - Same methodology as h_tests.py")
    print("\nVisualization Improvements:")
    print("  - Box plots with individual data points")
    print("  - Violin plots showing full distributions")
    print("  - Mean and median lines clearly marked")
    print("  - Sample sizes displayed for each education level")
    print("  - Correlation coefficients with significance shown on each plot")
    print("\nGenerated files:")
    print(f"  - {OUTPUT_FOLDER}/education_correlations.csv (correlation summary)")
    print(f"  - {OUTPUT_FOLDER}/edu_vs_rair.png (box plot)")
    print(f"  - {OUTPUT_FOLDER}/edu_vs_rsr.png (box plot)")
    print(f"  - {OUTPUT_FOLDER}/edu_vs_plausibility.png (box plot)")
    print(f"  - {OUTPUT_FOLDER}/edu_vs_conf_delta.png (box plot)")
    print(f"  - {OUTPUT_FOLDER}/edu_vs_accuracy.png (box plot)")
    print(f"  - {OUTPUT_FOLDER}/edu_metrics_by_level.png (multi-panel violin plots)")
    print(f"  - {OUTPUT_FOLDER}/edu_aor_scatter.png (AOR scatter: RAIR vs RSR)")
    print("\nRegression results are printed above (full statsmodels output)")
    print("\n")


if __name__ == "__main__":
    main()

