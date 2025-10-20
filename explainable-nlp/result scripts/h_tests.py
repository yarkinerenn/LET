"""
Hypothesis Testing Script with Cluster-Robust Standard Errors
==============================================================

This script implements a rigorous statistical analysis for WITHIN-SUBJECTS 
(repeated measures) design with proper accounting for within-participant correlation.

Study Design: WITHIN-SUBJECTS (Repeated Measures)
Each participant sees both faithful and unfaithful explanations (and both model sizes)
across different trials. Observations from the same participant are correlated.

Methodology:
1. NORMALITY TESTS: Shapiro-Wilk tests on all continuous variables
   - Tests overall and by groups (faith, model_size)
   
2. REGRESSION WITH CLUSTER-ROBUST STANDARD ERRORS:
   - Binary outcomes (RAIR, RSR, accuracy): Logistic regression with cluster-robust SEs
   - Continuous outcomes (confidence, plausibility): OLS with cluster-robust SEs
   - Clustering by participant accounts for within-subject correlation
   - Prevents underestimation of standard errors from repeated measures
   
3. WITHIN-SUBJECTS COMPARISONS (Participant-level, Paired):
   - Check normality of differences
   - If normal: Paired t-test with Cohen's d (paired) effect size
   - If non-normal: Wilcoxon signed-rank test with rank-biserial correlation
   
4. OUTPUTS:
   - Hypothesis test results with coefficients, p-values, and significance
   - Number of clusters (participants) for each test
   - Descriptive statistics by groups
   - Participant-level aggregates (RAIR, RSR, etc.)
   - Within-subjects (paired) comparisons with effect sizes
   - Summary tables exported to CSV

Key Improvement: Cluster-robust SEs properly account for repeated measures, 
providing valid inference when observations within participants are correlated.

Significance levels: * p<.05, ** p<.01, *** p<.001
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import shapiro, mannwhitneyu, ttest_ind
from plots import (
    plot_mean_rair_rsr_by_faith,
    plot_mean_conf_change_by_faith,
    plot_mean_final_accuracy_by_faith,
    plot_plausibility_violin_by_faith,
    plot_per_question_accuracy,
    plot_per_question_accuracy_by_modelsize,
    plot_per_question_accuracy_by_faithfulness,
    plot_aor_scatter_by_faith,
    plot_aor_scatter_by_modelsize,
    plot_conf_change_by_agreement,
    plot_conf_vs_rair_scatter,
    plot_conf_vs_rsr_scatter,
    plot_rair_rsr_by_modelsize,
    plot_conf_change_by_modelsize,
    plot_plaus_vs_rair_rsr,
    plot_accuracy_by_modelsize,
    plot_plausibility_by_modelsize,
    plot_confidence_plausibility_distribution,
    plot_human_accuracy_before_after
)

def logit(formula, data):
    model = smf.logit(formula, data=data.dropna()).fit(disp=False)
    return model

def ols(formula, data):
    model = smf.ols(formula, data=data.dropna()).fit()
    return model

def logit_clustered(formula, data, cluster_var='participant'):
    """
    Logistic regression with cluster-robust standard errors.
    Accounts for within-participant correlation in repeated measures.
    """
    clean_data = data.dropna()
    model = smf.logit(formula, data=clean_data)
    # Cluster-robust SEs by participant
    result = model.fit(disp=False, 
                      cov_type='cluster', 
                      cov_kwds={'groups': clean_data[cluster_var]})
    return result

def ols_clustered(formula, data, cluster_var='participant'):
    """
    OLS regression with cluster-robust standard errors.
    Accounts for within-participant correlation in repeated measures.
    """
    clean_data = data.dropna()
    model = smf.ols(formula, data=clean_data)
    # Cluster-robust SEs by participant
    result = model.fit(cov_type='cluster', 
                      cov_kwds={'groups': clean_data[cluster_var]})
    return result

def logit_robust(formula, data):
    """Logistic regression with robust standard errors (HC3) - DEPRECATED, use logit_clustered"""
    model = smf.logit(formula, data=data.dropna()).fit(disp=False, cov_type='HC3')
    return model

def ols_robust(formula, data):
    """OLS with robust standard errors (HC3) - DEPRECATED, use ols_clustered"""
    model = smf.ols(formula, data=data.dropna()).fit(cov_type='HC3')
    return model

def summarize(model):
    return {
        "params": model.params.to_dict(),
        "pvalues": model.pvalues.to_dict(),
        "conf_int": model.conf_int().to_dict(),
        "n": int(model.nobs)
    }

def check_normality(data, variable_name, alpha=0.05):
    """
    Perform Shapiro-Wilk normality test
    Returns: dict with test statistic, p-value, and whether data is normal
    """
    clean_data = data.dropna()
    if len(clean_data) < 3:
        return {"variable": variable_name, "n": len(clean_data), "normal": None, "reason": "insufficient data"}
    
    stat, p_value = shapiro(clean_data)
    is_normal = p_value > alpha
    
    return {
        "variable": variable_name,
        "n": len(clean_data),
        "statistic": stat,
        "p_value": p_value,
        "normal": is_normal,
        "alpha": alpha
    }

def descriptive_stats_by_group(long_df, variable, group_var='faith'):
    """
    Compute descriptive statistics by group (similar to Schemmer's approach)
    """
    groups = long_df.groupby(group_var)[variable].agg(['mean', 'std', 'count']).reset_index()
    groups.columns = [group_var, 'mean', 'std', 'n']
    return groups

def compute_participant_aggregates(long_df):
    """
    Compute participant-level aggregates for WITHIN-SUBJECTS design.
    Each participant sees both faithful and unfaithful trials, so we compute
    metrics separately for each condition WITHIN each participant.
    """
    participant_stats = []
    
    for participant_id in long_df['participant'].unique():
        p_data = long_df[long_df['participant'] == participant_id]
        
        # For within-subjects design, compute metrics for each condition
        for faith_val in [0, 1]:
            faith_data = p_data[p_data['faith'] == faith_val]
            
            if len(faith_data) == 0:
                continue
            
            # RAIR: proportion of changed_to_correct among RAIR-eligible trials
            rair_eligible = faith_data[(faith_data['ai_correct']==1) & (faith_data['human_pre_correct']==0)]
            rair = rair_eligible['changed_to_correct'].mean() if len(rair_eligible) > 0 else np.nan
            
            # RSR: proportion of stayed_correct among RSR-eligible trials  
            rsr_eligible = faith_data[(faith_data['ai_correct']==0) & (faith_data['human_pre_correct']==1)]
            rsr = rsr_eligible['stayed_correct'].mean() if len(rsr_eligible) > 0 else np.nan
            
            # Mean confidence change
            mean_delta_conf = faith_data['delta_conf'].mean()
            
            # Mean plausibility
            mean_plaus = faith_data['plaus'].mean()
            
            # Accuracy metrics
            human_pre_accuracy = faith_data['human_pre_correct'].mean()
            faith_data_with_post = faith_data.copy()
            faith_data_with_post['post_correct'] = (faith_data_with_post['post'] == faith_data_with_post['gt']).astype(int)
            post_accuracy = faith_data_with_post['post_correct'].mean()
            ai_accuracy = faith_data['ai_correct'].mean()
            
            participant_stats.append({
                'participant': participant_id,
                'faith': faith_val,
                'RAIR': rair,
                'RSR': rsr,
                'mean_delta_conf': mean_delta_conf,
                'mean_plaus': mean_plaus,
                'human_pre_accuracy': human_pre_accuracy,
                'post_accuracy': post_accuracy,
                'ai_accuracy': ai_accuracy,
                'n_trials': len(faith_data),
                'n_rair_eligible': len(rair_eligible),
                'n_rsr_eligible': len(rsr_eligible)
            })
        
        # Also compute for model size
        for size_val in [0, 1]:
            size_data = p_data[p_data['model_size'] == size_val]
            
            if len(size_data) == 0:
                continue
            
            # Store model size metrics separately
            rair_eligible = size_data[(size_data['ai_correct']==1) & (size_data['human_pre_correct']==0)]
            rair = rair_eligible['changed_to_correct'].mean() if len(rair_eligible) > 0 else np.nan
            
            rsr_eligible = size_data[(size_data['ai_correct']==0) & (size_data['human_pre_correct']==1)]
            rsr = rsr_eligible['stayed_correct'].mean() if len(rsr_eligible) > 0 else np.nan
            
            mean_delta_conf = size_data['delta_conf'].mean()
            mean_plaus = size_data['plaus'].mean()
            
            participant_stats.append({
                'participant': participant_id,
                'model_size': size_val,
                'RAIR': rair,
                'RSR': rsr,
                'mean_delta_conf': mean_delta_conf,
                'mean_plaus': mean_plaus,
                'n_trials': len(size_data),
                'n_rair_eligible': len(rair_eligible),
                'n_rsr_eligible': len(rsr_eligible)
            })
    
    return pd.DataFrame(participant_stats)

def within_subjects_test(data, variable, group_var='faith', alpha=0.05):
    """
    Perform within-subjects (PAIRED) comparison for repeated measures design.
    1. Check normality of differences
    2. Use paired t-test if normal, Wilcoxon signed-rank if not normal
    Returns test results with effect size
    """
    from scipy.stats import wilcoxon, ttest_rel
    
    # Pivot data to get paired observations
    pivot_data = data.pivot_table(index='participant', columns=group_var, values=variable, aggfunc='mean')
    
    groups = sorted([c for c in pivot_data.columns if not pd.isna(c)])
    
    if len(groups) != 2:
        return {"error": f"Expected 2 groups, found {len(groups)}"}
    
    # Get paired data (only participants with both conditions)
    paired_data = pivot_data.dropna()
    
    if len(paired_data) < 3:
        return {"error": f"Insufficient paired observations (n={len(paired_data)})"}
    
    group0_data = paired_data[groups[0]]
    group1_data = paired_data[groups[1]]
    differences = group1_data - group0_data
    
    # Check normality of differences
    norm_diff = check_normality(differences, f"{variable}_differences")
    is_normal = norm_diff.get('normal', False)
    
    # Compute descriptive stats
    mean0, std0 = group0_data.mean(), group0_data.std()
    mean1, std1 = group1_data.mean(), group1_data.std()
    mean_diff = differences.mean()
    std_diff = differences.std()
    
    # Choose test based on normality
    if is_normal:
        # Use paired t-test
        t_stat, p_value = ttest_rel(group0_data, group1_data)
        test_name = "Paired t-test"
        test_stat = t_stat
        # Cohen's d for paired samples
        effect_size = mean_diff / std_diff if std_diff > 0 else np.nan
        effect_size_name = "Cohen's d (paired)"
    else:
        # Use Wilcoxon signed-rank test (non-parametric)
        try:
            w_stat, p_value = wilcoxon(group0_data, group1_data, alternative='two-sided')
            test_name = "Wilcoxon signed-rank test"
            test_stat = w_stat
            # Rank-biserial correlation for paired data
            n = len(group0_data)
            effect_size = (test_stat / (n * (n + 1) / 2)) * 2 - 1
            effect_size_name = "Rank-biserial (paired)"
        except:
            return {"error": "Wilcoxon test failed (possibly zero differences)"}
    
    return {
        'variable': variable,
        'group_var': group_var,
        'test': test_name,
        'groups': {groups[0]: {'mean': mean0, 'std': std0, 'n': len(group0_data)},
                   groups[1]: {'mean': mean1, 'std': std1, 'n': len(group1_data)}},
        'statistic': test_stat,
        'p_value': p_value,
        'effect_size': effect_size,
        'effect_size_name': effect_size_name,
        'difference': mean_diff,
        'n_paired': len(paired_data),
        'normality': {'differences': norm_diff, 'normal': is_normal}
    }

def between_group_test(data, variable, group_var='faith', alpha=0.05):
    """
    Perform between-group comparison:
    1. Check normality for each group
    2. Use t-test if normal, Mann-Whitney U if not normal
    Returns test results with effect size
    """
    groups = data[group_var].unique()
    groups = sorted([g for g in groups if not pd.isna(g)])
    
    if len(groups) != 2:
        return {"error": f"Expected 2 groups, found {len(groups)}"}
    
    group0_data = data[data[group_var] == groups[0]][variable].dropna()
    group1_data = data[data[group_var] == groups[1]][variable].dropna()
    
    if len(group0_data) < 3 or len(group1_data) < 3:
        return {"error": "Insufficient data in one or more groups"}
    
    # Check normality for each group
    norm0 = check_normality(group0_data, f"{variable}_group{groups[0]}")
    norm1 = check_normality(group1_data, f"{variable}_group{groups[1]}")
    
    both_normal = norm0.get('normal', False) and norm1.get('normal', False)
    
    # Compute descriptive stats
    mean0, std0 = group0_data.mean(), group0_data.std()
    mean1, std1 = group1_data.mean(), group1_data.std()
    
    # Choose test based on normality
    if both_normal:
        # Use independent t-test
        t_stat, p_value = ttest_ind(group0_data, group1_data)
        test_name = "Independent t-test"
        test_stat = t_stat
        # Cohen's d effect size
        pooled_std = np.sqrt(((len(group0_data)-1)*std0**2 + (len(group1_data)-1)*std1**2) / 
                             (len(group0_data) + len(group1_data) - 2))
        effect_size = (mean1 - mean0) / pooled_std if pooled_std > 0 else np.nan
        effect_size_name = "Cohen's d"
    else:
        # Use Mann-Whitney U test (non-parametric)
        u_stat, p_value = mannwhitneyu(group0_data, group1_data, alternative='two-sided')
        test_name = "Mann-Whitney U test"
        test_stat = u_stat
        # Rank-biserial correlation as effect size
        n0, n1 = len(group0_data), len(group1_data)
        effect_size = 1 - (2*u_stat) / (n0 * n1)
        effect_size_name = "Rank-biserial correlation"
    
    return {
        'variable': variable,
        'group_var': group_var,
        'test': test_name,
        'groups': {groups[0]: {'mean': mean0, 'std': std0, 'n': len(group0_data)},
                   groups[1]: {'mean': mean1, 'std': std1, 'n': len(group1_data)}},
        'statistic': test_stat,
        'p_value': p_value,
        'effect_size': effect_size,
        'effect_size_name': effect_size_name,
        'difference': mean1 - mean0,
        'normality': {'group0': norm0, 'group1': norm1, 'both_normal': both_normal}
    }

def make_long(df_trials, n_trials=16):
    """
    Build long per-trial DataFrame with:
      participant, Q, pre, post, gt, ai, faith(F/U), plaus (numeric),
      delta_conf (post-pre), ai_correct, human_pre_correct,
      changed_to_correct, stayed_correct
    Assumes df_trials includes columns:
      Qn_Review, Qn_ReviewExp, Qn_GT, Qn_AI, Qn_Faith, Qn_Plausibility, Qn_Delta
    """
    rows = []
    for idx, row in df_trials.iterrows():
        for q in range(1, n_trials+1):
            pre  = str(row.get(f"Q{q}_Review", "")).strip()
            post = str(row.get(f"Q{q}_ReviewExp", "")).strip()
            gt   = str(row.get(f"Q{q}_GT", "")).strip().upper()
            ai   = str(row.get(f"Q{q}_AI", "")).strip().upper()
            faith= str(row.get(f"Q{q}_Faith", "")).strip().upper()  # 'F' or 'U'
            plaus = pd.to_numeric(row.get(f"Q{q}_Plausibility", np.nan), errors="coerce")
            dconf = pd.to_numeric(row.get(f"Q{q}_Delta", np.nan), errors="coerce")
            model_size = pd.to_numeric(row.get(f"Q{q}_Model_Size", np.nan), errors="coerce")

            def norm_dt(x):
                x = x.lower()
                if x in {"d","deceptive"}: return "D"
                if x in {"t","truthful"}:  return "T"
                return ""

            preN, postN = norm_dt(pre), norm_dt(post)

            valid = (preN in {"D","T"}) and (postN in {"D","T"}) and (gt in {"D","T"}) and (ai in {"D","T"})
            if not valid:
                continue

            ai_correct = int(ai == gt)
            human_pre_correct = int(preN == gt)
            changed_to_correct = int((preN != gt) and (postN == gt) and (ai == gt))
            stayed_correct = int((preN == gt) and (postN == gt) and (ai != gt))

            rows.append({
                "participant": idx,
                "Q": q,
                "pre": preN,
                "post": postN,
                "gt": gt,
                "ai": ai,
                "faith": 1 if faith == "F" else 0,  # 1=faithful, 0=unfaithful
                "plaus": plaus,
                "delta_conf": dconf,
                "model_size": model_size,  # 1=big LLM, 0=small LLM
                "ai_correct": ai_correct,
                "human_pre_correct": human_pre_correct,
                "changed_to_correct": changed_to_correct,
                "stayed_correct": stayed_correct
            })
    return pd.DataFrame(rows)

# H1: Faithfulness -> RAIR (among AI-correct & human initially wrong)
def test_H1(long_df, normality_results=None):
    df = long_df[(long_df["ai_correct"]==1) & (long_df["human_pre_correct"]==0)].copy()
    # DV: changed_to_correct (binary), IV: faith
    # Use cluster-robust SEs to account for repeated measures
    m = logit_clustered("changed_to_correct ~ faith", df, cluster_var='participant')
    result = summarize(m)
    result['test_type'] = 'Logistic Regression (Cluster-Robust SEs)'
    result['n_clusters'] = len(df['participant'].unique())
    return result

# H2: Faithfulness -> RSR (among AI-wrong & human initially correct)
def test_H2(long_df, normality_results=None):
    df = long_df[(long_df["ai_correct"]==0) & (long_df["human_pre_correct"]==1)].copy()
    m = logit_clustered("stayed_correct ~ faith", df, cluster_var='participant')
    result = summarize(m)
    result['test_type'] = 'Logistic Regression (Cluster-Robust SEs)'
    result['n_clusters'] = len(df['participant'].unique())
    return result

# H3: Faithfulness -> confidence calibration (participants' confidence better aligns with correctness)
def test_H3(long_df, normality_results=None):
    df = long_df.copy()
    # Use cluster-robust SEs to account for repeated measures
    m = ols_clustered("delta_conf ~ faith", df, cluster_var='participant')
    result = summarize(m)
    result['test_type'] = 'OLS (Cluster-Robust SEs)'
    result['n_clusters'] = len(df['participant'].unique())
    if normality_results and 'delta_conf' in normality_results:
        result['normality'] = normality_results['delta_conf']
    return result

# H4: Faithfulness affects final task accuracy (complementary team performance)
def test_H4(long_df, normality_results=None):
    df = long_df.copy()
    df["post_correct"] = (df["post"] == df["gt"]).astype(int)
    m = logit_clustered("post_correct ~ faith", df, cluster_var='participant')
    result = summarize(m)
    result['test_type'] = 'Logistic Regression (Cluster-Robust SEs)'
    result['n_clusters'] = len(df['participant'].unique())
    return result

# H5: Faithfulness increases perceived plausibility
def test_H5(long_df, normality_results=None):
    df = long_df.copy()
    m = ols_clustered("plaus ~ faith", df, cluster_var='participant')
    result = summarize(m)
    result['test_type'] = 'OLS (Cluster-Robust SEs)'
    result['n_clusters'] = len(df['participant'].unique())
    if normality_results and 'plaus' in normality_results:
        result['normality'] = normality_results['plaus']
    return result

# H6: Larger confidence changes predict higher RAIR (participants more often switch from wrong to correct)
def test_H6(long_df, normality_results=None):
    df = long_df[(long_df["ai_correct"]==1) & (long_df["human_pre_correct"]==0)].copy()
    m = logit_clustered("changed_to_correct ~ delta_conf", df, cluster_var='participant')
    result = summarize(m)
    result['test_type'] = 'Logistic Regression (Cluster-Robust SEs)'
    result['n_clusters'] = len(df['participant'].unique())
    return result

# H7: Smaller or negative confidence changes predict higher RSR (participants resist incorrect AI advice)
def test_H7(long_df, normality_results=None):
    df = long_df[(long_df["ai_correct"]==0) & (long_df["human_pre_correct"]==1)].copy()
    m = logit_clustered("stayed_correct ~ delta_conf", df, cluster_var='participant')
    result = summarize(m)
    result['test_type'] = 'Logistic Regression (Cluster-Robust SEs)'
    result['n_clusters'] = len(df['participant'].unique())
    return result

# H8: Larger LLMs produce more plausible explanations  (model_size: 1=large, 0=small)
def test_H8(long_df, normality_results=None):
    m = ols_clustered("plaus ~ model_size", long_df, cluster_var='participant')
    result = summarize(m)
    result['test_type'] = 'OLS (Cluster-Robust SEs)'
    result['n_clusters'] = len(long_df['participant'].unique())
    if normality_results and 'plaus' in normality_results:
        result['normality'] = normality_results['plaus']
    return result

# H9: Larger LLMs produce higher RAIR (on RAIR-eligible subset)
def test_H9(long_df, normality_results=None):
    df = long_df[(long_df["ai_correct"]==1) & (long_df["human_pre_correct"]==0)].copy()
    m = logit_clustered("changed_to_correct ~ model_size", df, cluster_var='participant')
    result = summarize(m)
    result['test_type'] = 'Logistic Regression (Cluster-Robust SEs)'
    result['n_clusters'] = len(df['participant'].unique())
    return result

# H10: Larger LLMs produce higher RSR (on RSR-eligible subset)
def test_H10(long_df, normality_results=None):
    df = long_df[(long_df["ai_correct"]==0) & (long_df["human_pre_correct"]==1)].copy()
    m = logit_clustered("stayed_correct ~ model_size", df, cluster_var='participant')
    result = summarize(m)
    result['test_type'] = 'Logistic Regression (Cluster-Robust SEs)'
    result['n_clusters'] = len(df['participant'].unique())
    return result

# H11: Larger LLMs produce bigger confidence changes
def test_H11(long_df, normality_results=None):
    m = ols_clustered("delta_conf ~ model_size", long_df, cluster_var='participant')
    result = summarize(m)
    result['test_type'] = 'OLS (Cluster-Robust SEs)'
    result['n_clusters'] = len(long_df['participant'].unique())
    if normality_results and 'delta_conf' in normality_results:
        result['normality'] = normality_results['delta_conf']
    return result

# H12: Larger LLMs lead to higher final task accuracy (complementary team performance)
def test_H12(long_df, normality_results=None):
    df = long_df.copy()
    df["post_correct"] = (df["post"] == df["gt"]).astype(int)
    m = logit_clustered("post_correct ~ model_size", df, cluster_var='participant')
    result = summarize(m)
    result['test_type'] = 'Logistic Regression (Cluster-Robust SEs)'
    result['n_clusters'] = len(df['participant'].unique())
    return result

# H13: Higher perceived plausibility is associated with higher RAIR
def test_H13(long_df, normality_results=None):
    df = long_df[(long_df["ai_correct"]==1) & (long_df["human_pre_correct"]==0)].copy()
    m = logit_clustered("changed_to_correct ~ plaus", df, cluster_var='participant')
    result = summarize(m)
    result['test_type'] = 'Logistic Regression (Cluster-Robust SEs)'
    result['n_clusters'] = len(df['participant'].unique())
    return result

# H14: Higher perceived plausibility is associated with higher RSR
def test_H14(long_df, normality_results=None):
    df = long_df[(long_df["ai_correct"]==0) & (long_df["human_pre_correct"]==1)].copy()
    m = logit_clustered("stayed_correct ~ plaus", df, cluster_var='participant')
    result = summarize(m)
    result['test_type'] = 'Logistic Regression (Cluster-Robust SEs)'
    result['n_clusters'] = len(df['participant'].unique())
    return result

def run_normality_tests(long_df):
    """
    Run Shapiro-Wilk normality tests on key continuous variables
    Similar to Schemmer et al.'s approach
    """
    normality_results = {}
    
    # Test continuous variables
    variables_to_test = ['delta_conf', 'plaus']
    
    for var in variables_to_test:
        if var in long_df.columns:
            normality_results[var] = check_normality(long_df[var], var)
    
    # Also test by groups (faithful vs unfaithful) for between-group comparisons
    for var in variables_to_test:
        if var in long_df.columns:
            for faith_val in [0, 1]:
                subset = long_df[long_df['faith'] == faith_val][var]
                faith_label = 'faithful' if faith_val == 1 else 'unfaithful'
                normality_results[f'{var}_{faith_label}'] = check_normality(
                    subset, f'{var} ({faith_label})'
                )
    
    # Test by model size
    for var in variables_to_test:
        if var in long_df.columns:
            for size_val in [0, 1]:
                subset = long_df[long_df['model_size'] == size_val][var]
                size_label = 'large' if size_val == 1 else 'small'
                normality_results[f'{var}_model_{size_label}'] = check_normality(
                    subset, f'{var} (model {size_label})'
                )
    
    return normality_results

def create_hypothesis_summary_table(results):
    """
    Create a summary table of hypothesis test results for easy interpretation
    """
    summary_rows = []
    
    hypothesis_descriptions = {
        "H1": "Faithfulness → RAIR",
        "H2": "Faithfulness → RSR", 
        "H3": "Faithfulness → Δ-Confidence",
        "H4": "Faithfulness → Final Accuracy",
        "H5": "Faithfulness → Plausibility",
        "H6": "Δ-Confidence → RAIR",
        "H7": "Δ-Confidence → RSR",
        "H8": "Model Size → Plausibility",
        "H9": "Model Size → RAIR",
        "H10": "Model Size → RSR",
        "H11": "Model Size → Δ-Confidence",
        "H12": "Model Size → Final Accuracy",
        "H13": "Plausibility → RAIR",
        "H14": "Plausibility → RSR"
    }
    
    for h_key, res in results.items():
        if 'error' not in res:
            params = res.get('params', {})
            pvalues = res.get('pvalues', {})
            
            # Get the main predictor (not intercept)
            predictor_keys = [k for k in params.keys() if k.lower() != 'intercept']
            if predictor_keys:
                predictor = predictor_keys[0]
                coef = params[predictor]
                p_val = pvalues[predictor]
                
                sig_level = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                supported = "Yes" if p_val < 0.05 else "No"
                
                summary_rows.append({
                    'Hypothesis': h_key,
                    'Relationship': hypothesis_descriptions.get(h_key, h_key),
                    'Test': res.get('test_type', 'N/A'),
                    'β': coef,
                    'p-value': p_val,
                    'Sig': sig_level,
                    'Supported': supported,
                    'N': res.get('n', 'N/A')
                })
    
    return pd.DataFrame(summary_rows)

def run_all_hypotheses(df_trials, n_trials=16):
    long_df = make_long(df_trials, n_trials=n_trials)
    
    # First, run normality tests
    print("\n" + "="*60)
    print("NORMALITY TESTS (Shapiro-Wilk)")
    print("="*60)
    normality_results = run_normality_tests(long_df)
    
    for key, result in normality_results.items():
        if result.get('normal') is not None:
            status = "NORMAL" if result['normal'] else "NOT NORMAL"
            print(f"{result['variable']:40} | W={result['statistic']:.4f}, p={result['p_value']:.4f} | {status}")
        else:
            print(f"{result['variable']:40} | {result.get('reason', 'N/A')}")
    
    print("\n" + "="*60)
    print("HYPOTHESIS TESTS (Cluster-Robust Standard Errors)")
    print("="*60)
    print("Note: Clustering by participant to account for repeated measures")
    print("="*60 + "\n")
    
    # Run hypothesis tests with robust methods
    results = {
        "H1":  test_H1(long_df, normality_results),
        "H2":  test_H2(long_df, normality_results),
        "H3":  test_H3(long_df, normality_results),
        "H4":  test_H4(long_df, normality_results),
        "H5":  test_H5(long_df, normality_results),
        "H6":  test_H6(long_df, normality_results),
        "H7":  test_H7(long_df, normality_results),
        "H8":  test_H8(long_df, normality_results),
        "H9":  test_H9(long_df, normality_results),
        "H10": test_H10(long_df, normality_results),
        "H11": test_H11(long_df, normality_results),
        "H12": test_H12(long_df, normality_results),
        "H13": test_H13(long_df, normality_results),
        "H14": test_H14(long_df, normality_results),
    }
    return results, long_df, normality_results

def main():
    df_trials = pd.read_excel("experiment_results_with_metrics.xlsx")

    results, long_df, normality_results = run_all_hypotheses(df_trials, n_trials=16)
    
    # Create and display summary table
    print("\n" + "="*60)
    print("HYPOTHESIS SUMMARY TABLE")
    print("="*60)
    summary_table = create_hypothesis_summary_table(results)
    print(summary_table.to_string(index=False))
    
    # Save summary table to CSV
    summary_table.to_csv("hypothesis_summary_table.csv", index=False)
    print("\n✓ Summary table saved to: hypothesis_summary_table.csv")

    print("\n" + "="*60)
    print("DETAILED HYPOTHESIS TEST RESULTS")
    print("="*60)
    for key, res in results.items():
        print(f"\n{key}:")
        if isinstance(res, dict) and "error" in res:
            print(f"  Error: {res['error']}")
        else:
            print(f"  Test Type: {res.get('test_type', 'N/A')}")
            params = res.get("params", {})
            pvalues = res.get("pvalues", {})
            conf_int = res.get("conf_int", {})
            
            print(f"  Coefficients:")
            for param_name, coef in params.items():
                p_val = pvalues.get(param_name, np.nan)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                print(f"    {param_name:20} β={coef:8.4f}, p={p_val:.4f} {sig}")
            
            print(f"  Number of observations: {res.get('n', 'N/A')}")
            if 'n_clusters' in res:
                print(f"  Number of clusters (participants): {res.get('n_clusters', 'N/A')}")
            
            # Show normality info if available
            if 'normality' in res:
                norm_info = res['normality']
                if norm_info.get('normal') is not None:
                    status = "NORMAL" if norm_info['normal'] else "NOT NORMAL"
                    print(f"  Normality: {status} (W={norm_info['statistic']:.4f}, p={norm_info['p_value']:.4f})")
    
    print("\n" + "="*60)
    print("DESCRIPTIVE STATISTICS BY GROUP")
    print("="*60)
    
    # Final accuracies by faithfulness
    print("\n--- Final Accuracy (Post-Explanation) by Faithfulness ---")
    df_with_post = long_df.dropna(subset=['post', 'gt', 'faith']).copy()
    df_with_post['post_correct'] = (df_with_post['post'] == df_with_post['gt']).astype(float)
    
    faithful_acc = df_with_post[df_with_post['faith'] == 1]['post_correct']
    unfaithful_acc = df_with_post[df_with_post['faith'] == 0]['post_correct']
    
    print(f"  Faithful:    M = {faithful_acc.mean():.4f}, SD = {faithful_acc.std():.4f}, n = {len(faithful_acc)}")
    print(f"  Unfaithful:  M = {unfaithful_acc.mean():.4f}, SD = {unfaithful_acc.std():.4f}, n = {len(unfaithful_acc)}")
    print(f"  Difference:  ΔM = {faithful_acc.mean() - unfaithful_acc.mean():.4f} (Faithful - Unfaithful)")
    
    # Descriptive stats for key variables by faith
    print("\n--- Delta Confidence by Faithfulness ---")
    print(descriptive_stats_by_group(long_df, 'delta_conf', 'faith'))
    
    print("\n--- Plausibility by Faithfulness ---")
    print(descriptive_stats_by_group(long_df, 'plaus', 'faith'))
    
    print("\n--- Delta Confidence by Model Size ---")
    print(descriptive_stats_by_group(long_df, 'delta_conf', 'model_size'))
    
    print("\n--- Plausibility by Model Size ---")
    print(descriptive_stats_by_group(long_df, 'plaus', 'model_size'))
    
    print("\n=== Long DataFrame Summary ===")
    print(f"Total observations: {len(long_df)}")
    print(f"\nDataFrame Description:")
    print(long_df.describe())
    
    # Participant-level aggregates (Schemmer-style between-subjects analysis)
    print("\n" + "="*60)
    print("PARTICIPANT-LEVEL AGGREGATES")
    print("="*60)
    
    participant_df = compute_participant_aggregates(long_df)
    print(f"\nTotal participants: {len(participant_df)}")
    print(f"\nParticipant-level summary:")
    print(participant_df.describe())
    
    # Save participant aggregates
    participant_df.to_csv("participant_level_aggregates.csv", index=False)
    print("\n✓ Participant aggregates saved to: participant_level_aggregates.csv")
    
    # Within-subjects comparisons (repeated measures design)
    print("\n" + "="*60)
    print("WITHIN-SUBJECTS COMPARISONS (Paired/Repeated Measures)")
    print("="*60)
    
    # Test by faithfulness
    variables_to_test = ['RAIR', 'RSR', 'mean_delta_conf', 'mean_plaus', 
                         'human_pre_accuracy', 'post_accuracy']
    
    print("\n--- Comparisons by FAITHFULNESS (Paired Tests) ---")
    faith_comparisons = []
    
    # Filter to only faith-related rows
    faith_df = participant_df[participant_df['faith'].notna()].copy()
    
    for var in variables_to_test:
        result = within_subjects_test(faith_df, var, 'faith')
        if 'error' in result:
            print(f"\n{var}: {result['error']}")
        else:
            print(f"\n{var}:")
            print(f"  Test: {result['test']} (N pairs = {result['n_paired']})")
            for g, stats in result['groups'].items():
                label = "Faithful" if g == 1 else "Unfaithful"
                print(f"  {label:12} | M={stats['mean']:.4f}, SD={stats['std']:.4f}")
            sig = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else "ns"
            print(f"  Statistic={result['statistic']:.4f}, p={result['p_value']:.4f} {sig}")
            print(f"  Mean Difference={result['difference']:.4f}")
            print(f"  {result['effect_size_name']}={result['effect_size']:.4f}")
            
            # Store for CSV export
            faith_comparisons.append({
                'Variable': var,
                'Test': result['test'],
                'N_Pairs': result['n_paired'],
                'Unfaithful_Mean': list(result['groups'].values())[0]['mean'],
                'Unfaithful_SD': list(result['groups'].values())[0]['std'],
                'Faithful_Mean': list(result['groups'].values())[1]['mean'],
                'Faithful_SD': list(result['groups'].values())[1]['std'],
                'Mean_Difference': result['difference'],
                'Statistic': result['statistic'],
                'p_value': result['p_value'],
                'Effect_Size': result['effect_size'],
                'Effect_Size_Type': result['effect_size_name'],
                'Significant': 'Yes' if result['p_value'] < 0.05 else 'No'
            })
    
    # Export within-subjects comparisons
    if faith_comparisons:
        faith_df_export = pd.DataFrame(faith_comparisons)
        faith_df_export.to_csv("within_subjects_comparisons_faithfulness.csv", index=False)
        print("\n✓ Faithfulness comparisons saved to: within_subjects_comparisons_faithfulness.csv")
    
    print("\n\n--- Comparisons by MODEL SIZE (Paired Tests) ---")
    model_size_comparisons = []
    
    # Filter to only model_size-related rows
    size_df = participant_df[participant_df['model_size'].notna()].copy()
    
    for var in variables_to_test:
        result = within_subjects_test(size_df, var, 'model_size')
        if 'error' in result:
            print(f"\n{var}: {result['error']}")
        else:
            print(f"\n{var}:")
            print(f"  Test: {result['test']} (N pairs = {result['n_paired']})")
            for g, stats in result['groups'].items():
                label = "Large LLM" if g == 1 else "Small LLM"
                print(f"  {label:12} | M={stats['mean']:.4f}, SD={stats['std']:.4f}")
            sig = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else "ns"
            print(f"  Statistic={result['statistic']:.4f}, p={result['p_value']:.4f} {sig}")
            print(f"  Mean Difference={result['difference']:.4f}")
            print(f"  {result['effect_size_name']}={result['effect_size']:.4f}")
            
            # Store for CSV export
            model_size_comparisons.append({
                'Variable': var,
                'Test': result['test'],
                'N_Pairs': result['n_paired'],
                'Small_LLM_Mean': list(result['groups'].values())[0]['mean'],
                'Small_LLM_SD': list(result['groups'].values())[0]['std'],
                'Large_LLM_Mean': list(result['groups'].values())[1]['mean'],
                'Large_LLM_SD': list(result['groups'].values())[1]['std'],
                'Mean_Difference': result['difference'],
                'Statistic': result['statistic'],
                'p_value': result['p_value'],
                'Effect_Size': result['effect_size'],
                'Effect_Size_Type': result['effect_size_name'],
                'Significant': 'Yes' if result['p_value'] < 0.05 else 'No'
            })
    
    # Export within-subjects comparisons
    if model_size_comparisons:
        model_size_df_export = pd.DataFrame(model_size_comparisons)
        model_size_df_export.to_csv("within_subjects_comparisons_modelsize.csv", index=False)
        print("\n✓ Model size comparisons saved to: within_subjects_comparisons_modelsize.csv")

    # Plots
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    plot_mean_rair_rsr_by_faith(long_df)
    plot_mean_conf_change_by_faith(long_df)
    plot_mean_final_accuracy_by_faith(long_df)
    plot_plausibility_violin_by_faith(long_df)
    plot_per_question_accuracy(long_df)
    plot_per_question_accuracy_by_modelsize(long_df)
    plot_per_question_accuracy_by_faithfulness(long_df)
    plot_human_accuracy_before_after(long_df)
    plot_confidence_plausibility_distribution(df_trials)
    plot_aor_scatter_by_faith(long_df)
    plot_aor_scatter_by_modelsize(long_df)
    plot_conf_change_by_agreement(long_df)
    plot_conf_vs_rair_scatter(long_df)
    plot_conf_vs_rsr_scatter(long_df)
    plot_rair_rsr_by_modelsize(long_df)
    plot_conf_change_by_modelsize(long_df)
    plot_accuracy_by_modelsize(long_df)
    plot_plaus_vs_rair_rsr(long_df)
    plot_plausibility_by_modelsize(long_df)
    print("✓ All visualizations generated")
    
    # Final summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE - OUTPUT FILES")
    print("="*60)
    print("\nCSV Files Generated:")
    print("  • hypothesis_summary_table.csv")
    print("  • participant_level_aggregates.csv")
    print("  • within_subjects_comparisons_faithfulness.csv")
    print("  • within_subjects_comparisons_modelsize.csv")
    print("\nAll visualization plots have been saved as PNG files.")
    print("\nNote: This analysis uses WITHIN-SUBJECTS (repeated measures) tests")
    print("because each participant experienced both conditions (faithful/unfaithful).")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
