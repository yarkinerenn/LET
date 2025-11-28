"""
NLP Experience Analysis Script
===============================

Analyzes the relationship between participants' self-rated NLP experience (1-5 Likert scale)
and key performance metrics using rigorous statistical methods.

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
- Box plots with overlaid individual data points (clearer for Likert scale)
- Violin plots showing full distribution shapes
- Mean and median lines clearly marked
- Sample sizes for each experience level
- Trend lines and correlation coefficients displayed

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
NLP_EXP_COL = config.DEMOGRAPHIC_COLUMNS["nlp_experience"]
DATA_FILE = config.PROCESSED_DATA_FILE
OUTPUT_FOLDER = config.PLOT_DIRS["nlp_experience"]


def add_nlp_experience_to_long(long_df, df_wide, nlp_col):
    """
    Add participant-level NLP experience rating to long format DataFrame.
    
    Parameters:
    -----------
    long_df : pd.DataFrame
        Long format data with 'participant' column
    df_wide : pd.DataFrame
        Wide format data with NLP experience column
    nlp_col : str
        Name of NLP experience column
        
    Returns:
    --------
    pd.DataFrame
        Long format with added 'nlp_experience' column
    """
    # Create mapping from participant index to NLP experience
    nlp_exp_map = df_wide[nlp_col].to_dict()
    long_df['nlp_experience'] = long_df['participant'].map(nlp_exp_map)
    return long_df


def compute_participant_level_metrics(long_df, df_wide):
    """
    Aggregate trial-level metrics to participant level and merge with wide format metrics.
    
    Parameters:
    -----------
    long_df : pd.DataFrame
        Long format data with trial-level metrics
    df_wide : pd.DataFrame
        Wide format data with participant-level metrics
        
    Returns:
    --------
    pd.DataFrame
        Participant-level aggregates with all metrics
    """
    # Aggregate from long format
    participant_agg = long_df.groupby('participant').agg({
        'plaus': 'mean',
        'delta_conf': 'mean',
        'nlp_experience': 'first'
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


def spearman_correlations(participant_df, nlp_col, metrics):
    """
    Compute Spearman rank correlations between NLP experience and all metrics.
    
    Parameters:
    -----------
    participant_df : pd.DataFrame
        Participant-level data
    nlp_col : str
        Name of NLP experience column
    metrics : list
        List of metric column names to correlate
        
    Returns:
    --------
    pd.DataFrame
        Correlation results with rho, p-value, n, and significance
    """
    results = []
    
    for metric in metrics:
        # Remove NaN values for this pair
        valid_data = participant_df[[nlp_col, metric]].dropna()
        
        if len(valid_data) < 3:
            # Not enough data for correlation
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
        rho, p_value = stats.spearmanr(valid_data[nlp_col], valid_data[metric])
        
        # Determine significance
        if p_value < 0.001:
            sig = '***'
        elif p_value < 0.01:
            sig = '**'
        elif p_value < 0.05:
            sig = '*'
        else:
            sig = ''
        
        # Interpret effect size (Cohen's guidelines for correlation)
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


def plot_boxplot_by_experience(df, x_col, y_col, title, output_path):
    """
    Create box plot with overlaid points showing distribution by NLP experience level.
    Much clearer than scatter plots for Likert scale data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data to plot
    x_col : str
        X-axis column (NLP experience)
    y_col : str
        Y-axis column (metric)
    title : str
        Plot title
    output_path : str
        Path to save the plot
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
    
    bp = ax.boxplot(box_data, positions=positions, widths=0.5, patch_artist=True,
                     boxprops=dict(facecolor=TUM_LIGHT_BLUE, alpha=0.7, edgecolor=TUM_BLUE_DARK),
                     whiskerprops=dict(color=TUM_BLUE_DARK, linewidth=1.5),
                     capprops=dict(color=TUM_BLUE_DARK, linewidth=1.5),
                     medianprops=dict(color=TUM_ORANGE, linewidth=2.5),
                     flierprops=dict(marker='o', markerfacecolor=TUM_GRAY_50, markersize=6, alpha=0.5))
    
    # Overlay individual points with jitter
    for pos in positions:
        pos_data = plot_data[plot_data[x_col] == pos][y_col].values
        jitter = np.random.normal(0, 0.08, len(pos_data))
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
    
    ax.set_xlabel('NLP Experience Level', fontsize=13, fontweight='bold')
    ax.set_ylabel(y_col.replace('_', ' '), fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim(0.5, 5.5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(['1\n(Novice)', '2', '3\n(Intermediate)', '4', '5\n(Expert)'])
    ax.grid(True, alpha=0.2, linestyle='--', axis='y')
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_metrics_by_experience(participant_df, output_path):
    """
    Create separate subplots for each metric showing clear distributions by experience level.
    
    Parameters:
    -----------
    participant_df : pd.DataFrame
        Participant-level data
    output_path : str
        Path to save the plot
    """
    metrics = ['RAIR_user', 'RSR_user', 'Mean_Plausibility', 'Mean_Conf_Delta', 'Final_Accuracy_User']
    metric_labels = ['RAIR', 'RSR', 'Mean Plausibility', 'Mean Confidence Change', 'Final Accuracy']
    colors = [TUM_BLUE, TUM_ORANGE, TUM_GREEN, TUM_MED_BLUE, TUM_BEIGE]
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        ax = axes[idx]
        plot_data = participant_df[['nlp_experience', metric]].dropna()
        
        if len(plot_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Create violin plot
        positions = sorted(plot_data['nlp_experience'].unique())
        violin_data = [plot_data[plot_data['nlp_experience'] == pos][metric].values for pos in positions]
        
        parts = ax.violinplot(violin_data, positions=positions, widths=0.6, 
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
            pos_data = plot_data[plot_data['nlp_experience'] == pos][metric].values
            jitter = np.random.normal(0, 0.06, len(pos_data))
            ax.scatter(pos + jitter, pos_data, alpha=0.3, s=25, color='black', edgecolors='none')
        
        # Add trend line
        z = np.polyfit(plot_data['nlp_experience'], plot_data[metric], 1)
        p = np.poly1d(z)
        x_line = np.linspace(1, 5, 100)
        ax.plot(x_line, p(x_line), color='red', linewidth=2, linestyle='--', alpha=0.7, label='Trend')
        
        # Compute correlation
        rho, p_value = stats.spearmanr(plot_data['nlp_experience'], plot_data[metric])
        sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        
        # Add stats annotation
        ax.text(0.02, 0.98, f'ρ = {rho:.3f} {sig}', transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('NLP Experience', fontsize=11, fontweight='bold')
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_xlim(0.5, 5.5)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.grid(True, alpha=0.2, linestyle='--', axis='y')
    
    # Remove the 6th subplot
    fig.delaxes(axes[5])
    
    plt.suptitle('Performance Metrics by NLP Experience Level', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_aor_scatter_by_nlp_experience(long_df, output_path):
    """
    Plot AOR scatter: RAIR (x-axis) vs RSR (y-axis) with one point per NLP experience level.
    AOR (Appropriateness of Reliance) = (RAIR + RSR) / 2
    
    Parameters:
    -----------
    long_df : pd.DataFrame
        Long format data with nlp_experience column
    output_path : str
        Path to save the plot
    """
    # RAIR-eligible: AI correct & human initially wrong
    df_rair = long_df[(long_df["ai_correct"]==1) & (long_df["human_pre_correct"]==0)].copy()
    df_rair["rair"] = df_rair["changed_to_correct"].astype(float)
    rair_by_exp = df_rair.groupby("nlp_experience")["rair"].mean().rename("RAIR")
    
    # RSR-eligible: AI wrong & human initially correct
    df_rsr = long_df[(long_df["ai_correct"]==0) & (long_df["human_pre_correct"]==1)].copy()
    df_rsr["rsr"] = df_rsr["stayed_correct"].astype(float)
    rsr_by_exp = df_rsr.groupby("nlp_experience")["rsr"].mean().rename("RSR")
    
    # Combine
    summary = pd.concat([rair_by_exp, rsr_by_exp], axis=1).reset_index()
    summary = summary.dropna()  # Remove any experience levels with missing data
    summary["AOR"] = (summary["RAIR"] + summary["RSR"]) / 2.0
    summary["exp_label"] = summary["nlp_experience"].apply(lambda x: f"Level {int(x)}")
    
    if len(summary) == 0:
        print("Warning: No valid data for AOR plot")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Color gradient for experience levels
    colors = [TUM_BLUE_DARKER, TUM_BLUE_DARK, TUM_BLUE, TUM_MED_BLUE, TUM_LIGHT_BLUE]
    exp_levels = sorted(summary["nlp_experience"].unique())
    color_map = {level: colors[i % len(colors)] for i, level in enumerate(exp_levels)}
    point_colors = [color_map[exp] for exp in summary["nlp_experience"]]
    
    # Scatter plot
    scatter = ax.scatter(summary["RAIR"], summary["RSR"], s=250, c=point_colors, 
                        edgecolor="black", linewidth=2, zorder=10, alpha=0.8)
    
    # Annotate each point with level and AOR
    for _, row in summary.iterrows():
        level = int(row["nlp_experience"])
        label = f"Level {level}\nAOR={row['AOR']:.3f}"
        
        # Position labels to avoid overlap
        if level <= 2:
            xytext = (10, -15)
            va = 'top'
        elif level == 3:
            xytext = (10, 10)
            va = 'bottom'
        else:
            xytext = (-10, 10)
            va = 'bottom'
        
        ax.annotate(label, (row["RAIR"], row["RSR"]), 
                   xytext=xytext, textcoords="offset points",
                   ha="left" if level <= 3 else "right", va=va,
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="white", 
                            edgecolor=color_map[row["nlp_experience"]], alpha=0.95, linewidth=2))
    
    # Add diagonal line for reference (AOR = 0.5 line where RAIR = RSR)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1.5, label='RAIR = RSR')
    
    # Formatting
    ax.set_xlabel('RAIR (Reliance on AI when Right)', fontsize=13, fontweight='bold')
    ax.set_ylabel('RSR (Resistance to wrong AI)', fontsize=13, fontweight='bold')
    ax.set_title('Appropriateness of Reliance (AOR) by NLP Experience\nRAIR vs RSR', 
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
                             label=f'Level {int(level)}',
                             markerfacecolor=color_map[level], 
                             markeredgecolor='black', markersize=10)
                      for level in exp_levels]
    ax.legend(handles=legend_elements, loc='lower right', title='NLP Experience', 
             framealpha=0.95, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\n" + "="*80)
    print("AOR (Appropriateness of Reliance) by NLP Experience Level")
    print("="*80)
    print(f"{'Level':<8} {'RAIR':<10} {'RSR':<10} {'AOR':<10}")
    print("-"*80)
    for _, row in summary.iterrows():
        print(f"{int(row['nlp_experience']):<8} {row['RAIR']:<10.3f} {row['RSR']:<10.3f} {row['AOR']:<10.3f}")
    print("="*80)
    
    print(f"\nSaved AOR plot to: {output_path}")


def regression_analysis_clustered(long_df, nlp_col='nlp_experience'):
    """
    Perform regression analyses with cluster-robust standard errors.
    Uses the same methodology as h_tests.py to account for repeated measures.
    
    Parameters:
    -----------
    long_df : pd.DataFrame
        Long format data with trial-level observations
    nlp_col : str
        Name of NLP experience column
        
    Returns:
    --------
    dict
        Dictionary with regression results for each outcome
    """
    results = {}
    
    # Prepare data - ensure nlp_experience is numeric
    analysis_df = long_df.copy()
    analysis_df[nlp_col] = pd.to_numeric(analysis_df[nlp_col], errors='coerce')
    
    print("\n" + "="*80)
    print("REGRESSION ANALYSES WITH CLUSTER-ROBUST STANDARD ERRORS")
    print("="*80)
    print("Clustering by participant to account for repeated measures")
    print("Same methodology as h_tests.py")
    
    # 1. Plausibility (continuous) - OLS with cluster-robust SEs
    print("\n" + "-"*80)
    print("1. PLAUSIBILITY ~ NLP_Experience (OLS with cluster-robust SEs)")
    print("-"*80)
    try:
        model_plaus = ols_clustered(f'plaus ~ {nlp_col}', data=analysis_df, cluster_var='participant')
        results['plausibility'] = model_plaus
        print(model_plaus.summary())
        coef = model_plaus.params[nlp_col]
        pval = model_plaus.pvalues[nlp_col]
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        print(f"\nCoefficient: {coef:.4f} {sig}")
        print(f"Interpretation: Each 1-point increase in NLP experience is associated with")
        print(f"               {abs(coef):.4f} {'increase' if coef > 0 else 'decrease'} in plausibility rating")
    except Exception as e:
        print(f"Error: {e}")
        results['plausibility'] = None
    
    # 2. Confidence change (continuous) - OLS with cluster-robust SEs
    print("\n" + "-"*80)
    print("2. CONFIDENCE_CHANGE ~ NLP_Experience (OLS with cluster-robust SEs)")
    print("-"*80)
    try:
        model_conf = ols_clustered(f'delta_conf ~ {nlp_col}', data=analysis_df, cluster_var='participant')
        results['confidence_change'] = model_conf
        print(model_conf.summary())
        coef = model_conf.params[nlp_col]
        pval = model_conf.pvalues[nlp_col]
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        print(f"\nCoefficient: {coef:.4f} {sig}")
        print(f"Interpretation: Each 1-point increase in NLP experience is associated with")
        print(f"               {abs(coef):.4f} {'increase' if coef > 0 else 'decrease'} in confidence change")
    except Exception as e:
        print(f"Error: {e}")
        results['confidence_change'] = None
    
    # 3. Human final accuracy (binary) - Logistic regression with cluster-robust SEs
    print("\n" + "-"*80)
    print("3. FINAL_ACCURACY (post==gt) ~ NLP_Experience (Logistic with cluster-robust SEs)")
    print("-"*80)
    try:
        analysis_df['final_correct'] = (analysis_df['post'] == analysis_df['gt']).astype(int)
        model_acc = logit_clustered(f'final_correct ~ {nlp_col}', data=analysis_df, cluster_var='participant')
        results['final_accuracy'] = model_acc
        print(model_acc.summary())
        coef = model_acc.params[nlp_col]
        pval = model_acc.pvalues[nlp_col]
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        odds_ratio = np.exp(coef)
        print(f"\nLog-odds coefficient: {coef:.4f} {sig}")
        print(f"Odds ratio: {odds_ratio:.4f}")
        print(f"Interpretation: Each 1-point increase in NLP experience multiplies the odds")
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
        if len(rair_df) >= 10:  # Need enough observations
            model_rair = logit_clustered(f'changed_to_correct ~ {nlp_col}', data=rair_df, cluster_var='participant')
            results['rair'] = model_rair
            print(model_rair.summary())
            coef = model_rair.params[nlp_col]
            pval = model_rair.pvalues[nlp_col]
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            odds_ratio = np.exp(coef)
            print(f"\nLog-odds coefficient: {coef:.4f} {sig}")
            print(f"Odds ratio: {odds_ratio:.4f}")
            print(f"Interpretation: Each 1-point increase in NLP experience multiplies the odds")
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
        if len(rsr_df) >= 10:  # Need enough observations
            model_rsr = logit_clustered(f'stayed_correct ~ {nlp_col}', data=rsr_df, cluster_var='participant')
            results['rsr'] = model_rsr
            print(model_rsr.summary())
            coef = model_rsr.params[nlp_col]
            pval = model_rsr.pvalues[nlp_col]
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            odds_ratio = np.exp(coef)
            print(f"\nLog-odds coefficient: {coef:.4f} {sig}")
            print(f"Odds ratio: {odds_ratio:.4f}")
            print(f"Interpretation: Each 1-point increase in NLP experience multiplies the odds")
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
    """
    Save correlation results to CSV file.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Correlation results
    output_path : str
        Path to save CSV
    """
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved correlation summary to: {output_path}")


def print_correlation_summary(results_df):
    """
    Print formatted correlation summary to console.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Correlation results
    """
    print("\n" + "="*80)
    print("NLP EXPERIENCE CORRELATION ANALYSIS")
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
    """
    Main analysis function.
    """
    print("="*80)
    print("NLP EXPERIENCE ANALYSIS")
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
    
    # Check if NLP experience column exists
    if NLP_EXP_COL not in df_wide.columns:
        print(f"\nError: Column '{NLP_EXP_COL}' not found in data.")
        print("Available columns containing 'nlp' or 'experience':")
        relevant_cols = [c for c in df_wide.columns if 'nlp' in c.lower() or 'experience' in c.lower()]
        for col in relevant_cols:
            print(f"  - {col}")
        return
    
    # Create long format
    print("\nConverting to long format...")
    long_df = make_long(df_wide, n_trials=config.N_TRIALS)
    print(f"Created {len(long_df)} trial observations")
    
    # Add NLP experience to long format
    long_df = add_nlp_experience_to_long(long_df, df_wide, NLP_EXP_COL)
    
    # Compute participant-level metrics
    print("\nComputing participant-level metrics...")
    participant_metrics = compute_participant_level_metrics(long_df, df_wide)
    print(f"Aggregated data for {len(participant_metrics)} participants")
    
    # Check NLP experience distribution
    nlp_exp_dist = participant_metrics['nlp_experience'].value_counts().sort_index()
    print("\nNLP Experience Distribution:")
    for level, count in nlp_exp_dist.items():
        print(f"  Level {int(level)}: {count} participants")
    
    # Define metrics to analyze
    metrics = ['RAIR_user', 'RSR_user', 'Mean_Plausibility', 'Mean_Conf_Delta', 'Final_Accuracy_User']
    
    # Perform Spearman correlations (exploratory analysis)
    print("\n" + "="*80)
    print("PART 1: SPEARMAN CORRELATIONS (Exploratory)")
    print("="*80)
    print("Performing Spearman rank correlations on participant-level aggregates...")
    correlation_results = spearman_correlations(participant_metrics, 'nlp_experience', metrics)
    
    # Print summary
    print_correlation_summary(correlation_results)
    
    # Save correlation summary
    csv_path = os.path.join(OUTPUT_FOLDER, 'nlp_experience_correlations.csv')
    save_correlation_summary(correlation_results, csv_path)
    
    # Perform regression analyses with cluster-robust SEs (main analysis)
    print("\n" + "="*80)
    print("PART 2: REGRESSION ANALYSES (Main Analysis)")
    print("="*80)
    print("Using trial-level data with cluster-robust SEs (same as h_tests.py)")
    regression_results = regression_analysis_clustered(long_df, nlp_col='nlp_experience')
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Individual box plots (clearer than scatter plots for Likert scale)
    plot_configs = [
        ('RAIR_user', 'Reliance on AI when AI is Right (RAIR) by NLP Experience', 'nlp_exp_vs_rair.png'),
        ('RSR_user', 'Resistance to Wrong AI (RSR) by NLP Experience', 'nlp_exp_vs_rsr.png'),
        ('Mean_Plausibility', 'Mean Plausibility Rating by NLP Experience', 'nlp_exp_vs_plausibility.png'),
        ('Mean_Conf_Delta', 'Mean Confidence Change by NLP Experience', 'nlp_exp_vs_conf_delta.png'),
        ('Final_Accuracy_User', 'Final Accuracy by NLP Experience', 'nlp_exp_vs_accuracy.png')
    ]
    
    for metric, title, filename in plot_configs:
        plot_path = os.path.join(OUTPUT_FOLDER, filename)
        plot_boxplot_by_experience(participant_metrics, 'nlp_experience', metric, title, plot_path)
    
    # Multi-panel plot
    grouped_plot_path = os.path.join(OUTPUT_FOLDER, 'nlp_exp_metrics_by_level.png')
    plot_metrics_by_experience(participant_metrics, grouped_plot_path)
    
    # AOR scatter plot
    print("\nCreating AOR scatter plot...")
    aor_plot_path = os.path.join(OUTPUT_FOLDER, 'nlp_exp_aor_scatter.png')
    plot_aor_scatter_by_nlp_experience(long_df, aor_plot_path)
    
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
    print("  - Box plots with individual data points (clearer than scatter plots)")
    print("  - Violin plots showing full distributions")
    print("  - Mean and median lines clearly marked")
    print("  - Sample sizes displayed for each experience level")
    print("  - Correlation coefficients with significance shown on each plot")
    print("\nGenerated files:")
    print(f"  - {OUTPUT_FOLDER}/nlp_experience_correlations.csv (correlation summary)")
    print(f"  - {OUTPUT_FOLDER}/nlp_exp_vs_rair.png (box plot)")
    print(f"  - {OUTPUT_FOLDER}/nlp_exp_vs_rsr.png (box plot)")
    print(f"  - {OUTPUT_FOLDER}/nlp_exp_vs_plausibility.png (box plot)")
    print(f"  - {OUTPUT_FOLDER}/nlp_exp_vs_conf_delta.png (box plot)")
    print(f"  - {OUTPUT_FOLDER}/nlp_exp_vs_accuracy.png (box plot)")
    print(f"  - {OUTPUT_FOLDER}/nlp_exp_metrics_by_level.png (multi-panel violin plots)")
    print(f"  - {OUTPUT_FOLDER}/nlp_exp_aor_scatter.png (AOR scatter: RAIR vs RSR)")
    print("\nRegression results are printed above (full statsmodels output)")
    print("\n")


if __name__ == "__main__":
    main()

