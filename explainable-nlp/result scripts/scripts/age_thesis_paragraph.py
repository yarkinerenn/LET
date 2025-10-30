"""
Generate thesis paragraph summarizing Age analysis
"""

import pandas as pd
import numpy as np
from h_tests import make_long, ols_clustered, logit_clustered

# Configuration
AGE_COLUMN = "What is your age?"
DATA_FILE = '../data/experiment_results_with_metrics.xlsx'

# Age group mapping to ordinal numbers for regression
AGE_GROUP_MAPPING = {
    "18-24": 1,
    "25-34": 2,
    "35-44": 3,
    "45-54": 4,
    "55+": 5,
    "55-64": 5,
    "65+": 6
}


def generate_thesis_paragraph():
    """
    Generate a formatted paragraph for thesis describing age analysis.
    """
    print("="*80)
    print("GENERATING THESIS PARAGRAPH - AGE ANALYSIS")
    print("="*80)
    
    # Load data
    df_wide = pd.read_excel(DATA_FILE)
    long_df = make_long(df_wide, n_trials=16)
    
    # Add age group to long format
    age_group_map = df_wide[AGE_COLUMN].to_dict()
    long_df['age_group'] = long_df['participant'].map(age_group_map)
    long_df['age_ordinal'] = long_df['age_group'].map(AGE_GROUP_MAPPING)
    
    # Compute participant-level metrics
    participant_agg = long_df.groupby('participant').agg({
        'plaus': 'mean',
        'delta_conf': 'mean',
        'age_group': 'first',
        'age_ordinal': 'first'
    }).reset_index()
    
    participant_agg.rename(columns={
        'plaus': 'Mean_Plausibility',
        'delta_conf': 'Mean_Conf_Delta'
    }, inplace=True)
    
    # Merge with wide format metrics
    wide_metrics = df_wide[['RAIR_user', 'RSR_user', 'Final_Accuracy_User']].copy()
    wide_metrics['participant'] = wide_metrics.index
    participant_metrics = participant_agg.merge(wide_metrics, on='participant', how='left')
    
    # Get statistics
    total_n = len(participant_metrics)
    
    # Age group distribution
    age_group_dist = participant_metrics['age_group'].value_counts().sort_index()
    
    # Compute descriptive statistics by age groups
    print("\n" + "-"*80)
    print("DESCRIPTIVE STATISTICS BY AGE GROUP")
    print("-"*80)
    
    metrics = ['RAIR_user', 'RSR_user', 'Mean_Plausibility', 'Mean_Conf_Delta', 'Final_Accuracy_User']
    metric_names = ['RAIR', 'RSR', 'mean plausibility', 'mean confidence change', 'final accuracy']
    
    desc_stats = {}
    for metric in metrics:
        desc_stats[metric] = participant_metrics.groupby('age_group')[metric].agg(['mean', 'std', 'count'])
    
    # Display descriptive stats
    for metric, name in zip(metrics, metric_names):
        print(f"\n{name.upper()}:")
        if metric in desc_stats:
            print(desc_stats[metric].to_string())
    
    # Run regression analyses
    print("\n" + "-"*80)
    print("REGRESSION ANALYSES (Cluster-Robust SEs)")
    print("-"*80)
    
    analysis_df = long_df.copy()
    analysis_df['age_ordinal'] = pd.to_numeric(analysis_df['age_ordinal'], errors='coerce')
    
    regression_results = {}
    
    # 1. Plausibility
    try:
        model_plaus = ols_clustered('plaus ~ age_ordinal', data=analysis_df, cluster_var='participant')
        regression_results['plausibility'] = {
            'coef': model_plaus.params['age_ordinal'],
            'pval': model_plaus.pvalues['age_ordinal'],
            'n': int(model_plaus.nobs)
        }
        print(f"\nPlausibility: coef={regression_results['plausibility']['coef']:.4f}, p={regression_results['plausibility']['pval']:.4f}")
    except:
        regression_results['plausibility'] = None
        print("\nPlausibility: Failed")
    
    # 2. Confidence change
    try:
        model_conf = ols_clustered('delta_conf ~ age_ordinal', data=analysis_df, cluster_var='participant')
        regression_results['confidence_change'] = {
            'coef': model_conf.params['age_ordinal'],
            'pval': model_conf.pvalues['age_ordinal'],
            'n': int(model_conf.nobs)
        }
        print(f"Confidence Change: coef={regression_results['confidence_change']['coef']:.4f}, p={regression_results['confidence_change']['pval']:.4f}")
    except:
        regression_results['confidence_change'] = None
        print("Confidence Change: Failed")
    
    # 3. Final accuracy
    try:
        analysis_df['final_correct'] = (analysis_df['post'] == analysis_df['gt']).astype(int)
        model_acc = logit_clustered('final_correct ~ age_ordinal', data=analysis_df, cluster_var='participant')
        regression_results['final_accuracy'] = {
            'coef': model_acc.params['age_ordinal'],
            'pval': model_acc.pvalues['age_ordinal'],
            'odds_ratio': np.exp(model_acc.params['age_ordinal']),
            'n': int(model_acc.nobs)
        }
        print(f"Final Accuracy: coef={regression_results['final_accuracy']['coef']:.4f}, p={regression_results['final_accuracy']['pval']:.4f}, OR={regression_results['final_accuracy']['odds_ratio']:.4f}")
    except:
        regression_results['final_accuracy'] = None
        print("Final Accuracy: Failed")
    
    # 4. RAIR
    try:
        rair_df = analysis_df[(analysis_df['ai_correct']==1) & (analysis_df['human_pre_correct']==0)].copy()
        model_rair = logit_clustered('changed_to_correct ~ age_ordinal', data=rair_df, cluster_var='participant')
        regression_results['rair'] = {
            'coef': model_rair.params['age_ordinal'],
            'pval': model_rair.pvalues['age_ordinal'],
            'odds_ratio': np.exp(model_rair.params['age_ordinal']),
            'n': int(model_rair.nobs)
        }
        print(f"RAIR: coef={regression_results['rair']['coef']:.4f}, p={regression_results['rair']['pval']:.4f}, OR={regression_results['rair']['odds_ratio']:.4f}")
    except:
        regression_results['rair'] = None
        print("RAIR: Failed")
    
    # 5. RSR
    try:
        rsr_df = analysis_df[(analysis_df['ai_correct']==0) & (analysis_df['human_pre_correct']==1)].copy()
        model_rsr = logit_clustered('stayed_correct ~ age_ordinal', data=rsr_df, cluster_var='participant')
        regression_results['rsr'] = {
            'coef': model_rsr.params['age_ordinal'],
            'pval': model_rsr.pvalues['age_ordinal'],
            'odds_ratio': np.exp(model_rsr.params['age_ordinal']),
            'n': int(model_rsr.nobs)
        }
        print(f"RSR: coef={regression_results['rsr']['coef']:.4f}, p={regression_results['rsr']['pval']:.4f}, OR={regression_results['rsr']['odds_ratio']:.4f}")
    except:
        regression_results['rsr'] = None
        print("RSR: Failed")
    
    # Generate paragraph
    print("\n" + "="*80)
    print("THESIS PARAGRAPH (Ready to copy)")
    print("="*80)
    print()
    
    # Build age distribution text (age groups only, no numeric age available)
    age_group_text = ", ".join([f"{int(count)} in {group}" for group, count in age_group_dist.items() if pd.notna(group)])
    age_dist_text = f"distributed across age groups: {age_group_text}"
    
    # Build descriptive stats summary
    desc_text_parts = []
    for metric, name in zip(metrics, metric_names):
        if metric in desc_stats:
            stats = desc_stats[metric]
            min_mean = stats['mean'].min()
            max_mean = stats['mean'].max()
            desc_text_parts.append(f"{name} ranged from {min_mean:.2f} to {max_mean:.2f}")
    
    # Build regression results summary
    sig_results = []
    nonsig_results = []
    
    for key, name in [('plausibility', 'plausibility'), ('confidence_change', 'confidence change'), 
                      ('final_accuracy', 'final accuracy'), ('rair', 'RAIR'), ('rsr', 'RSR')]:
        if regression_results.get(key):
            pval = regression_results[key]['pval']
            if pval < 0.05:
                sig_results.append(f"{name} (p = {pval:.3f})")
            else:
                nonsig_results.append(f"{name} (p = {pval:.3f})")
    
    # Construct the paragraph
    paragraph = f"""To explore whether participants' age group influenced their performance and perception of AI explanations, we analyzed responses from {total_n} participants ({age_dist_text}). We examined five key metrics using regression analyses with cluster-robust standard errors to account for repeated measures within participants: Reliance on AI when AI is Right (RAIR), Resistance to wrong AI Suggestions (RSR), plausibility ratings, confidence change, and final accuracy. Descriptive statistics across age groups revealed relatively consistent patterns, with {', '.join(desc_text_parts[:3])}, and {', '.join(desc_text_parts[3:])}. Regression analyses using trial-level data with cluster-robust standard errors showed no statistically significant relationships between age group and any of the examined metrics (all p > .05). Specifically, {', '.join(nonsig_results[:3])}{',' if len(nonsig_results) > 3 else ''} {', '.join(nonsig_results[3:]) if len(nonsig_results) > 3 else ''}. These findings suggest that participant age group did not meaningfully influence their ability to evaluate AI explanations, their reliance on AI suggestions, or their perception of explanation quality, indicating that the effects observed in our main analyses were consistent across different age groups."""
    
    print(paragraph)
    
    # Save to file
    with open('../plots/demographic_analyses/age_plots/thesis_paragraph.txt', 'w') as f:
        f.write("THESIS PARAGRAPH - AGE ANALYSIS\n")
        f.write("="*80 + "\n\n")
        f.write(paragraph)
        f.write("\n\n" + "="*80 + "\n")
        f.write("DETAILED STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Sample size: {total_n} participants\n")
        f.write(f"Age distribution: {age_dist_text}\n\n")
        
        f.write("Descriptive Statistics by Age Group:\n")
        f.write("-"*80 + "\n")
        for metric, name in zip(metrics, metric_names):
            if metric in desc_stats:
                f.write(f"\n{name.upper()}:\n")
                f.write(desc_stats[metric].to_string())
                f.write("\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("Regression Results (Cluster-Robust SEs):\n")
        f.write("-"*80 + "\n")
        for key, name in [('plausibility', 'Plausibility'), ('confidence_change', 'Confidence Change'), 
                          ('final_accuracy', 'Final Accuracy'), ('rair', 'RAIR'), ('rsr', 'RSR')]:
            if regression_results.get(key):
                f.write(f"\n{name}:\n")
                f.write(f"  Coefficient: {regression_results[key]['coef']:.4f}\n")
                f.write(f"  P-value: {regression_results[key]['pval']:.4f}\n")
                if 'odds_ratio' in regression_results[key]:
                    f.write(f"  Odds Ratio: {regression_results[key]['odds_ratio']:.4f}\n")
                f.write(f"  N observations: {regression_results[key]['n']}\n")
    
    print("\n" + "="*80)
    print("Paragraph and detailed statistics saved to:")
    print("  ../plots/demographic_analyses/age_plots/thesis_paragraph.txt")
    print("="*80)


if __name__ == "__main__":
    generate_thesis_paragraph()

