"""
Gender Analysis Script
======================

Analyzes the relationship between participants' gender and key performance metrics
using rigorous statistical methods.

Metrics analyzed:
- RAIR (Reliance on AI when AI is right)
- RSR (Resistance to wrong AI)
- Plausibility ratings
- Confidence change
- Final accuracy

Statistical Methods:
1. Descriptive statistics by gender group
2. Regression with cluster-robust standard errors (main analysis, trial-level data)
   - OLS for continuous outcomes (plausibility, confidence change)
   - Logistic regression for binary outcomes (accuracy, RAIR, RSR)
   - Clustering by participant accounts for repeated measures
   - Same methodology as h_tests.py
   - Gender coded as dummy variable (Male as reference)

Visualizations:
- Box plots showing distribution by gender
- Violin plots showing full distribution shapes
- AOR scatter plot (RAIR vs RSR) by gender

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
GENDER_COLUMN = "What is your gender?"
DATA_FILE = '../data/experiment_results_with_metrics.xlsx'
OUTPUT_FOLDER = '../plots/demographic_analyses/gender_plots'


def add_gender_to_long(long_df, df_wide, gender_col):
    """
    Add participant-level gender to long format DataFrame.
    """
    # Create mapping from participant index to gender
    gender_map = df_wide[gender_col].to_dict()
    long_df['gender'] = long_df['participant'].map(gender_map)
    
    # Create dummy variable for regression (Male = 0, Female = 1, Other = 2)
    # Standardize gender values
    long_df['gender_clean'] = long_df['gender'].str.strip().str.lower()
    long_df['is_female'] = (long_df['gender_clean'] == 'female').astype(int)
    
    return long_df


def compute_participant_level_metrics(long_df, df_wide):
    """
    Aggregate trial-level metrics to participant level and merge with wide format metrics.
    """
    # Aggregate from long format
    participant_agg = long_df.groupby('participant').agg({
        'plaus': 'mean',
        'delta_conf': 'mean',
        'gender': 'first',
        'is_female': 'first'
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


def plot_boxplot_by_gender(df, y_col, title, output_path):
    """
    Create box plot with gender on x-axis.
    """
    # Remove NaN values
    plot_data = df[['gender', y_col]].dropna()
    
    if len(plot_data) == 0:
        print(f"Warning: No valid data for {title}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get unique genders and sort
    genders = sorted(plot_data['gender'].unique())
    positions = list(range(len(genders)))
    
    # Create box plot
    box_data = [plot_data[plot_data['gender'] == g][y_col].values for g in genders]
    
    bp = ax.boxplot(box_data, positions=positions, widths=0.5, patch_artist=True,
                     boxprops=dict(facecolor=TUM_LIGHT_BLUE, alpha=0.7, edgecolor=TUM_BLUE_DARK),
                     whiskerprops=dict(color=TUM_BLUE_DARK, linewidth=1.5),
                     capprops=dict(color=TUM_BLUE_DARK, linewidth=1.5),
                     medianprops=dict(color=TUM_ORANGE, linewidth=2.5),
                     flierprops=dict(marker='o', markerfacecolor=TUM_GRAY_50, markersize=6, alpha=0.5))
    
    # Overlay individual points with jitter
    for i, gender in enumerate(genders):
        gender_data = plot_data[plot_data['gender'] == gender][y_col].values
        jitter = np.random.normal(0, 0.08, len(gender_data))
        ax.scatter([i]*len(gender_data) + jitter, gender_data, alpha=0.4, s=40, color=TUM_BLUE, edgecolors='none')
    
    # Add mean line
    means = [plot_data[plot_data['gender'] == g][y_col].mean() for g in genders]
    ax.plot(positions, means, color=TUM_GREEN, linewidth=2.5, marker='D', markersize=8, 
            label='Mean', linestyle='--', zorder=10)
    
    # Add sample size annotations
    for i, gender in enumerate(genders):
        n = len(plot_data[plot_data['gender'] == gender])
        ax.text(i, ax.get_ylim()[1] * 0.98, f'n={n}', 
                ha='center', va='top', fontsize=9, color=TUM_GRAY_80)
    
    ax.set_xlabel('Gender', fontsize=13, fontweight='bold')
    ax.set_ylabel(y_col.replace('_', ' '), fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(positions)
    ax.set_xticklabels(genders)
    ax.grid(True, alpha=0.2, linestyle='--', axis='y')
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_metrics_by_gender(participant_df, output_path):
    """
    Create separate subplots for each metric showing distributions by gender.
    """
    metrics = ['RAIR_user', 'RSR_user', 'Mean_Plausibility', 'Mean_Conf_Delta', 'Final_Accuracy_User']
    metric_labels = ['RAIR', 'RSR', 'Mean Plausibility', 'Mean Confidence Change', 'Final Accuracy']
    colors = [TUM_BLUE, TUM_ORANGE, TUM_GREEN, TUM_MED_BLUE, TUM_BEIGE]
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        ax = axes[idx]
        plot_data = participant_df[['gender', metric]].dropna()
        
        if len(plot_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Get unique genders
        genders = sorted(plot_data['gender'].unique())
        positions = list(range(len(genders)))
        
        # Create violin plot
        violin_data = [plot_data[plot_data['gender'] == g][metric].values for g in genders]
        
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
        for i, gender in enumerate(genders):
            gender_data = plot_data[plot_data['gender'] == gender][metric].values
            jitter = np.random.normal(0, 0.06, len(gender_data))
            ax.scatter([i]*len(gender_data) + jitter, gender_data, alpha=0.3, s=25, color='black', edgecolors='none')
        
        ax.set_xlabel('Gender', fontsize=11, fontweight='bold')
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels(genders)
        ax.grid(True, alpha=0.2, linestyle='--', axis='y')
    
    # Remove the 6th subplot
    fig.delaxes(axes[5])
    
    plt.suptitle('Performance Metrics by Gender', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_aor_scatter_by_gender(long_df, output_path):
    """
    Plot AOR scatter: RAIR (x-axis) vs RSR (y-axis) with one point per gender.
    AOR (Appropriateness of Reliance) = (RAIR + RSR) / 2
    """
    # RAIR-eligible: AI correct & human initially wrong
    df_rair = long_df[(long_df["ai_correct"]==1) & (long_df["human_pre_correct"]==0)].copy()
    df_rair["rair"] = df_rair["changed_to_correct"].astype(float)
    rair_by_gender = df_rair.groupby("gender")["rair"].mean().rename("RAIR")
    
    # RSR-eligible: AI wrong & human initially correct
    df_rsr = long_df[(long_df["ai_correct"]==0) & (long_df["human_pre_correct"]==1)].copy()
    df_rsr["rsr"] = df_rsr["stayed_correct"].astype(float)
    rsr_by_gender = df_rsr.groupby("gender")["rsr"].mean().rename("RSR")
    
    # Combine
    summary = pd.concat([rair_by_gender, rsr_by_gender], axis=1).reset_index()
    summary = summary.dropna()  # Remove any gender with missing data
    summary["AOR"] = (summary["RAIR"] + summary["RSR"]) / 2.0
    
    if len(summary) == 0:
        print("Warning: No valid data for AOR plot")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Colors for gender (use different colors)
    gender_colors = {
        'Male': TUM_BLUE,
        'Female': TUM_ORANGE,
        'Non-binary': TUM_GREEN,
        'Prefer not to say': TUM_GRAY_50
    }
    
    # Get colors for each gender in summary
    point_colors = [gender_colors.get(g, TUM_MED_BLUE) for g in summary['gender']]
    
    # Scatter plot
    scatter = ax.scatter(summary["RAIR"], summary["RSR"], s=350, c=point_colors, 
                        edgecolor="black", linewidth=2, zorder=10, alpha=0.8)
    
    # Annotate each point with gender and AOR
    for idx, row in summary.iterrows():
        label = f"{row['gender']}\nAOR={row['AOR']:.3f}"
        
        # Position labels to avoid overlap
        if idx == 0:
            xytext = (10, -15)
            va = 'top'
            ha = 'left'
        else:
            xytext = (10, 10)
            va = 'bottom'
            ha = 'left'
        
        ax.annotate(label, (row["RAIR"], row["RSR"]), 
                   xytext=xytext, textcoords="offset points",
                   ha=ha, va=va,
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="white", 
                            edgecolor=gender_colors.get(row['gender'], TUM_MED_BLUE), 
                            alpha=0.95, linewidth=2))
    
    # Add diagonal line for reference
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1.5, label='RAIR = RSR')
    
    # Formatting
    ax.set_xlabel('RAIR (Reliance on AI when Right)', fontsize=13, fontweight='bold')
    ax.set_ylabel('RSR (Resistance to wrong AI)', fontsize=13, fontweight='bold')
    ax.set_title('Appropriateness of Reliance (AOR) by Gender\nRAIR vs RSR', 
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
                             label=row['gender'],
                             markerfacecolor=gender_colors.get(row['gender'], TUM_MED_BLUE), 
                             markeredgecolor='black', markersize=12)
                      for _, row in summary.iterrows()]
    ax.legend(handles=legend_elements, loc='lower right', title='Gender', 
             framealpha=0.95, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print("\n" + "="*80)
    print("AOR (Appropriateness of Reliance) by Gender")
    print("="*80)
    print(f"{'Gender':<20} {'RAIR':<10} {'RSR':<10} {'AOR':<10}")
    print("-"*80)
    for _, row in summary.iterrows():
        print(f"{row['gender']:<20} {row['RAIR']:<10.3f} {row['RSR']:<10.3f} {row['AOR']:<10.3f}")
    print("="*80)
    
    print(f"\nSaved AOR plot to: {output_path}")


def regression_analysis_clustered(long_df, gender_var='is_female'):
    """
    Perform regression analyses with cluster-robust standard errors.
    Uses the same methodology as h_tests.py to account for repeated measures.
    Gender is coded as dummy variable (Male = 0, Female = 1).
    """
    results = {}
    
    # Prepare data
    analysis_df = long_df.copy()
    
    print("\n" + "="*80)
    print("REGRESSION ANALYSES WITH CLUSTER-ROBUST STANDARD ERRORS")
    print("="*80)
    print("Clustering by participant to account for repeated measures")
    print("Same methodology as h_tests.py")
    print("Gender coded as: Male = 0 (reference), Female = 1")
    
    # 1. Plausibility (continuous) - OLS with cluster-robust SEs
    print("\n" + "-"*80)
    print("1. PLAUSIBILITY ~ Gender (OLS with cluster-robust SEs)")
    print("-"*80)
    try:
        model_plaus = ols_clustered(f'plaus ~ {gender_var}', data=analysis_df, cluster_var='participant')
        results['plausibility'] = model_plaus
        print(model_plaus.summary())
        coef = model_plaus.params[gender_var]
        pval = model_plaus.pvalues[gender_var]
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        print(f"\nCoefficient: {coef:.4f} {sig}")
        print(f"Interpretation: Females have {abs(coef):.4f} {'higher' if coef > 0 else 'lower'} plausibility ratings than males")
    except Exception as e:
        print(f"Error: {e}")
        results['plausibility'] = None
    
    # 2. Confidence change (continuous) - OLS with cluster-robust SEs
    print("\n" + "-"*80)
    print("2. CONFIDENCE_CHANGE ~ Gender (OLS with cluster-robust SEs)")
    print("-"*80)
    try:
        model_conf = ols_clustered(f'delta_conf ~ {gender_var}', data=analysis_df, cluster_var='participant')
        results['confidence_change'] = model_conf
        print(model_conf.summary())
        coef = model_conf.params[gender_var]
        pval = model_conf.pvalues[gender_var]
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        print(f"\nCoefficient: {coef:.4f} {sig}")
        print(f"Interpretation: Females have {abs(coef):.4f} {'higher' if coef > 0 else 'lower'} confidence change than males")
    except Exception as e:
        print(f"Error: {e}")
        results['confidence_change'] = None
    
    # 3. Human final accuracy (binary) - Logistic regression with cluster-robust SEs
    print("\n" + "-"*80)
    print("3. FINAL_ACCURACY (post==gt) ~ Gender (Logistic with cluster-robust SEs)")
    print("-"*80)
    try:
        analysis_df['final_correct'] = (analysis_df['post'] == analysis_df['gt']).astype(int)
        model_acc = logit_clustered(f'final_correct ~ {gender_var}', data=analysis_df, cluster_var='participant')
        results['final_accuracy'] = model_acc
        print(model_acc.summary())
        coef = model_acc.params[gender_var]
        pval = model_acc.pvalues[gender_var]
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        odds_ratio = np.exp(coef)
        print(f"\nLog-odds coefficient: {coef:.4f} {sig}")
        print(f"Odds ratio: {odds_ratio:.4f}")
        print(f"Interpretation: Females have {odds_ratio:.4f}x the odds of being correct compared to males")
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
            model_rair = logit_clustered(f'changed_to_correct ~ {gender_var}', data=rair_df, cluster_var='participant')
            results['rair'] = model_rair
            print(model_rair.summary())
            coef = model_rair.params[gender_var]
            pval = model_rair.pvalues[gender_var]
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            odds_ratio = np.exp(coef)
            print(f"\nLog-odds coefficient: {coef:.4f} {sig}")
            print(f"Odds ratio: {odds_ratio:.4f}")
            print(f"Interpretation: Females have {odds_ratio:.4f}x the odds of accepting AI's correct suggestion")
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
            model_rsr = logit_clustered(f'stayed_correct ~ {gender_var}', data=rsr_df, cluster_var='participant')
            results['rsr'] = model_rsr
            print(model_rsr.summary())
            coef = model_rsr.params[gender_var]
            pval = model_rsr.pvalues[gender_var]
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            odds_ratio = np.exp(coef)
            print(f"\nLog-odds coefficient: {coef:.4f} {sig}")
            print(f"Odds ratio: {odds_ratio:.4f}")
            print(f"Interpretation: Females have {odds_ratio:.4f}x the odds of resisting AI's wrong suggestion")
        else:
            print("Insufficient RSR-eligible observations for regression")
            results['rsr'] = None
    except Exception as e:
        print(f"Error: {e}")
        results['rsr'] = None
    
    print("\n" + "="*80)
    
    return results


def save_stats_summary(results_df, output_path):
    """Save descriptive statistics to CSV file."""
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved statistics summary to: {output_path}")


def main():
    """Main analysis function."""
    print("="*80)
    print("GENDER ANALYSIS")
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
    
    # Check if gender column exists
    if GENDER_COLUMN not in df_wide.columns:
        print(f"\nError: Column '{GENDER_COLUMN}' not found in data.")
        print("Available columns containing 'gender':")
        relevant_cols = [c for c in df_wide.columns if 'gender' in c.lower()]
        for col in relevant_cols:
            print(f"  - {col}")
        return
    
    # Create long format
    print("\nConverting to long format...")
    long_df = make_long(df_wide, n_trials=16)
    print(f"Created {len(long_df)} trial observations")
    
    # Add gender to long format
    long_df = add_gender_to_long(long_df, df_wide, GENDER_COLUMN)
    
    # Compute participant-level metrics
    print("\nComputing participant-level metrics...")
    participant_metrics = compute_participant_level_metrics(long_df, df_wide)
    print(f"Aggregated data for {len(participant_metrics)} participants")
    
    # Check gender distribution
    gender_dist = participant_metrics['gender'].value_counts().sort_index()
    print("\nGender Distribution:")
    for gender, count in gender_dist.items():
        if pd.notna(gender):
            print(f"  {gender}: {count} participants")
    
    # Define metrics to analyze
    metrics = ['RAIR_user', 'RSR_user', 'Mean_Plausibility', 'Mean_Conf_Delta', 'Final_Accuracy_User']
    
    # Compute descriptive statistics by gender
    print("\n" + "="*80)
    print("DESCRIPTIVE STATISTICS BY GENDER")
    print("="*80)
    
    desc_stats = {}
    for metric in metrics:
        desc_stats[metric] = participant_metrics.groupby('gender')[metric].agg(['mean', 'std', 'count'])
        print(f"\n{metric}:")
        print(desc_stats[metric].to_string())
    
    # Perform regression analyses with cluster-robust SEs (main analysis)
    print("\n" + "="*80)
    print("REGRESSION ANALYSES (Main Analysis)")
    print("="*80)
    print("Using trial-level data with cluster-robust SEs (same as h_tests.py)")
    regression_results = regression_analysis_clustered(long_df, gender_var='is_female')
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Box plots
    boxplot_configs = [
        ('RAIR_user', 'Reliance on AI when AI is Right (RAIR) by Gender', 'gender_vs_rair.png'),
        ('RSR_user', 'Resistance to Wrong AI (RSR) by Gender', 'gender_vs_rsr.png'),
        ('Mean_Plausibility', 'Mean Plausibility Rating by Gender', 'gender_vs_plausibility.png'),
        ('Mean_Conf_Delta', 'Mean Confidence Change by Gender', 'gender_vs_conf_delta.png'),
        ('Final_Accuracy_User', 'Final Accuracy by Gender', 'gender_vs_accuracy.png')
    ]
    
    for metric, title, filename in boxplot_configs:
        plot_path = os.path.join(OUTPUT_FOLDER, filename)
        plot_boxplot_by_gender(participant_metrics, metric, title, plot_path)
    
    # Multi-panel plot
    grouped_plot_path = os.path.join(OUTPUT_FOLDER, 'gender_metrics_by_group.png')
    plot_metrics_by_gender(participant_metrics, grouped_plot_path)
    
    # AOR scatter plot
    print("\nCreating AOR scatter plot...")
    aor_plot_path = os.path.join(OUTPUT_FOLDER, 'gender_aor_scatter.png')
    plot_aor_scatter_by_gender(long_df, aor_plot_path)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll files saved in folder: {OUTPUT_FOLDER}/")
    print("\nStatistical Analyses Performed:")
    print("  1. Descriptive statistics by gender group")
    print("  2. Regression with cluster-robust SEs (main analysis, trial-level)")
    print("     - Accounts for repeated measures within participants")
    print("     - Same methodology as h_tests.py")
    print("     - Gender as dummy variable (Male = reference)")
    print("\nVisualization Types:")
    print("  - Box plots by gender")
    print("  - Multi-panel violin plots")
    print("  - AOR scatter plot by gender")
    print("\nGenerated files:")
    print(f"  - {OUTPUT_FOLDER}/gender_vs_rair.png (box plot)")
    print(f"  - {OUTPUT_FOLDER}/gender_vs_rsr.png (box plot)")
    print(f"  - {OUTPUT_FOLDER}/gender_vs_plausibility.png (box plot)")
    print(f"  - {OUTPUT_FOLDER}/gender_vs_conf_delta.png (box plot)")
    print(f"  - {OUTPUT_FOLDER}/gender_vs_accuracy.png (box plot)")
    print(f"  - {OUTPUT_FOLDER}/gender_metrics_by_group.png (multi-panel violin plots)")
    print(f"  - {OUTPUT_FOLDER}/gender_aor_scatter.png (AOR scatter: RAIR vs RSR)")
    print("\nRegression results are printed above (full statsmodels output)")
    print("\n")


if __name__ == "__main__":
    main()

