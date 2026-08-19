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
   - Hypothesis test results with coefficients, 95% CIs, p-values, and significance
   - Odds ratios with 95% CIs for all logistic models
   - Bonferroni-corrected significance flags (16 tests)
   - Number of clusters (participants) for each test
   - Descriptive statistics by groups
   - Participant-level aggregates (RAIR, RSR, etc.)
   - Within-subjects (paired) comparisons with effect sizes
   - Optional recruitment-source robustness check
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
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from plots import (
    plot_mean_rair_rsr_by_faith,
    plot_mean_conf_change_by_faith,
    plot_mean_final_accuracy_by_faith,
    plot_plausibility_violin_by_faith,
    plot_per_question_accuracy,
    plot_per_question_accuracy_by_modelsize,
    plot_per_question_accuracy_by_faithfulness,
    plot_rair_rsr_per_question,
    plot_aor_scatter_by_faith,
    plot_aor_scatter_by_modelsize,
    plot_plausibility_vs_accuracy,
    plot_plausibility_vs_conf_change,
    plot_plausibility_by_agreement,
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

# Item-level mean plausibility: for each question Q, the mean rating given by all
# participants who saw it. Built in make_long() with groupby("Q") -- see the
# assertion there.
#
# USE IT ONLY WHERE PLAUSIBILITY IS THE PREDICTOR (H13, H14, H15). Using an item
# mean as a predictor of the participant's own behaviour is the point: it replaces
# the participant's own rating, which is given after their revised decision, with a
# property of the explanation itself.
#
# DO NOT USE IT AS THE OUTCOME (H5, H8, H16). `faith` is fixed per item, so an
# item-level outcome regressed on an item-level predictor is constant within Q:
# every participant contributes an identical block of 16 rows, the participant-
# clustered covariance estimate collapses, and the SE comes out as exactly 0.0000.
# Those three hypotheses use the individual rating `plaus` as the outcome.
MEAN_PLAUSIBILITY_COLUMN = "mean_plausibility"
INDIVIDUAL_PLAUSIBILITY_COLUMN = "plaus"

# Leave-one-out item mean: for each trial, the mean rating that item received from
# every OTHER participant. This is what H13/H14/H15 actually use as the predictor.
# It answers the objection "your predictor still contains the participant's own
# rating" outright, rather than conceding a 1-in-102 contamination. Numerically it
# is within a few percent of MEAN_PLAUSIBILITY_COLUMN and changes no conclusion.
LOO_PLAUSIBILITY_COLUMN = "plaus_loo"

# ============================================================================
# CONFIGURATION
# ============================================================================

N_HYPOTHESES = 16              # used for the Bonferroni threshold
ALPHA = 0.05
BONFERRONI_ALPHA = ALPHA / N_HYPOTHESES

# Column in the raw trials file identifying where a participant was recruited.
# Set to the actual column name to enable the recruitment-source robustness check
# (paper Limitations: "do the main effects hold in the Prolific subsample?").
# Leave as None if the study did not record this.
RECRUITMENT_COLUMN = None      # e.g. "Recruitment Source" or "source"
PROLIFIC_LABEL = "Prolific"    # value in that column marking panel participants


def logit(formula, data):
    model = smf.logit(formula, data=data.dropna()).fit(disp=False)
    return model

def ols(formula, data):
    model = smf.ols(formula, data=data.dropna()).fit()
    return model

def _cluster_groups(clean_data, cluster_var):
    """
    Accept a single cluster variable ('participant') or a list for two-way
    clustering (['participant', 'Q']). Two-way is needed because `faith` is fixed
    per item and the plausibility aggregates take only 16 distinct values, so
    participant-only clustering treats 16 explanations as if they were 1632
    independent observations.
    """
    if isinstance(cluster_var, (list, tuple)):
        return clean_data[list(cluster_var)]
    return clean_data[cluster_var]


def logit_clustered(formula, data, cluster_var='participant'):
    """
    Logistic regression with cluster-robust standard errors.
    Accounts for within-participant correlation in repeated measures.
    `cluster_var` may be a list for two-way clustering.
    """
    clean_data = data.dropna()
    model = smf.logit(formula, data=clean_data)
    # Cluster-robust SEs by participant
    result = model.fit(disp=False,
                      cov_type='cluster',
                      cov_kwds={'groups': _cluster_groups(clean_data, cluster_var)})
    return result

def ols_clustered(formula, data, cluster_var='participant'):
    """
    OLS regression with cluster-robust standard errors.
    Accounts for within-participant correlation in repeated measures.
    `cluster_var` may be a list for two-way clustering.
    """
    clean_data = data.dropna()
    model = smf.ols(formula, data=clean_data)
    # Cluster-robust SEs by participant
    result = model.fit(cov_type='cluster',
                      cov_kwds={'groups': _cluster_groups(clean_data, cluster_var)})
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
    """
    Extract coefficients, p-values, standard errors and 95% confidence intervals.

    conf_int is stored as a nested dict keyed {0: {param: lower}, 1: {param: upper}},
    matching pandas' DataFrame.to_dict() layout for the two CI columns.
    """
    return {
        "params": model.params.to_dict(),
        "pvalues": model.pvalues.to_dict(),
        "bse": model.bse.to_dict(),
        "conf_int": model.conf_int().to_dict(),
        "n": int(model.nobs)
    }


def get_ci(res, param_name):
    """
    Pull the (lower, upper) 95% CI for one parameter out of a summarize() dict.
    Handles both integer and string column keys defensively.
    """
    ci = res.get("conf_int", {})
    if not ci:
        return (np.nan, np.nan)
    keys = list(ci.keys())
    if len(keys) < 2:
        return (np.nan, np.nan)
    lo_key, hi_key = keys[0], keys[1]
    lower = ci[lo_key].get(param_name, np.nan)
    upper = ci[hi_key].get(param_name, np.nan)
    return (lower, upper)


def is_logistic(res):
    """True if this result came from a logistic model (so odds ratios apply)."""
    return 'logistic' in str(res.get('test_type', '')).lower()


def get_main_predictor(res):
    """Return the name of the non-intercept predictor, or None."""
    params = res.get('params', {})
    predictors = [k for k in params.keys() if k.lower() != 'intercept']
    return predictors[0] if predictors else None


def sig_stars(p_val):
    return "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"


def format_ci(lower, upper, decimals=3):
    if pd.isna(lower) or pd.isna(upper):
        return "N/A"
    return f"[{lower:.{decimals}f}, {upper:.{decimals}f}]"

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

            # Mean plausibility across the questions in this condition.
            mean_plaus = faith_data[INDIVIDUAL_PLAUSIBILITY_COLUMN].mean()

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
            mean_plaus = size_data[INDIVIDUAL_PLAUSIBILITY_COLUMN].mean()

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

    # 95% CI on the mean paired difference (t-based)
    n_pairs = len(paired_data)
    if n_pairs > 1 and std_diff > 0:
        se_diff = std_diff / np.sqrt(n_pairs)
        t_crit = stats.t.ppf(1 - alpha / 2, df=n_pairs - 1)
        diff_ci = (mean_diff - t_crit * se_diff, mean_diff + t_crit * se_diff)
    else:
        diff_ci = (np.nan, np.nan)

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
        'difference_ci': diff_ci,
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
      mean_plausibility (mean rating for Q across participants),
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
    long_df = pd.DataFrame(rows)
    # Item-level mean plausibility: mean rating for each question Q across all
    # participants who rated it. Grouped by the item id (Q) and nothing else --
    # never by faith or model_size, which are the predictors in H5/H8.
    long_df[MEAN_PLAUSIBILITY_COLUMN] = long_df.groupby("Q")["plaus"].transform("mean")
    assert long_df[MEAN_PLAUSIBILITY_COLUMN].nunique() == long_df["Q"].nunique(), \
        "mean_plausibility must have one value per question"

    # Leave-one-out item mean: (sum of this item's ratings - own rating) / (n - 1).
    # Grouped by Q like the column above, but the focal participant's own rating is
    # removed, so it cannot restate their own decision. Unlike mean_plausibility it
    # deliberately varies within an item -- that is the whole point.
    grp = long_df.groupby("Q")["plaus"]
    n_ratings = grp.transform("count")
    total = grp.transform("sum")
    long_df[LOO_PLAUSIBILITY_COLUMN] = (total - long_df["plaus"]) / (n_ratings - 1)
    assert long_df.groupby("Q")[LOO_PLAUSIBILITY_COLUMN].nunique().gt(1).any(), \
        "plaus_loo must vary within a question; the subtraction did not take effect"
    return long_df

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
    m = ols_clustered(f"{INDIVIDUAL_PLAUSIBILITY_COLUMN} ~ faith", df, cluster_var='participant')
    result = summarize(m)
    result['test_type'] = 'OLS (Cluster-Robust SEs)'
    result['n_clusters'] = len(df['participant'].unique())
    if normality_results and INDIVIDUAL_PLAUSIBILITY_COLUMN in normality_results:
        result['normality'] = normality_results[INDIVIDUAL_PLAUSIBILITY_COLUMN]
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
    m = ols_clustered(f"{INDIVIDUAL_PLAUSIBILITY_COLUMN} ~ model_size", long_df, cluster_var='participant')
    result = summarize(m)
    result['test_type'] = 'OLS (Cluster-Robust SEs)'
    result['n_clusters'] = len(long_df['participant'].unique())
    if normality_results and INDIVIDUAL_PLAUSIBILITY_COLUMN in normality_results:
        result['normality'] = normality_results[INDIVIDUAL_PLAUSIBILITY_COLUMN]
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
    m = logit_clustered(f"changed_to_correct ~ {LOO_PLAUSIBILITY_COLUMN}", df, cluster_var='participant')
    result = summarize(m)
    result['test_type'] = 'Logistic Regression (Cluster-Robust SEs)'
    result['n_clusters'] = len(df['participant'].unique())
    return result

# H14: Higher perceived plausibility is associated with lower RSR (people are less resistant to incorrect AI advice when it's plausible)
def test_H14(long_df, normality_results=None):
    df = long_df[(long_df["ai_correct"]==0) & (long_df["human_pre_correct"]==1)].copy()
    m = logit_clustered(f"stayed_correct ~ {LOO_PLAUSIBILITY_COLUMN}", df, cluster_var='participant')
    result = summarize(m)
    result['test_type'] = 'Logistic Regression (Cluster-Robust SEs)'
    result['n_clusters'] = len(df['participant'].unique())
    return result

# H15: Higher perceived plausibility is associated with larger confidence changes
def test_H15(long_df, normality_results=None):
    df = long_df.dropna(subset=['delta_conf', LOO_PLAUSIBILITY_COLUMN]).copy()
    m = ols_clustered(f"delta_conf ~ {LOO_PLAUSIBILITY_COLUMN}", df, cluster_var='participant')
    result = summarize(m)
    result['test_type'] = 'OLS (Cluster-Robust SEs)'
    result['n_clusters'] = len(df['participant'].unique())
    if normality_results and 'delta_conf' in normality_results:
        result['normality'] = normality_results['delta_conf']
    return result

# H16: Explanations are rated as more plausible when human and AI initially agree
def test_H16(long_df, normality_results=None):
    df = long_df.dropna(subset=[INDIVIDUAL_PLAUSIBILITY_COLUMN, 'pre', 'ai']).copy()
    # Create agreement variable: 1 if human and AI agree initially, 0 if they disagree
    df["agreement"] = (df["pre"] == df["ai"]).astype(int)
    m = ols_clustered(f"{INDIVIDUAL_PLAUSIBILITY_COLUMN} ~ agreement", df, cluster_var='participant')
    result = summarize(m)
    result['test_type'] = 'OLS (Cluster-Robust SEs)'
    result['n_clusters'] = len(df['participant'].unique())
    if normality_results and INDIVIDUAL_PLAUSIBILITY_COLUMN in normality_results:
        result['normality'] = normality_results[INDIVIDUAL_PLAUSIBILITY_COLUMN]
    return result


# ============================================================================
# ADDITIONAL REPORTING: agreement descriptives, plausibility diagnosticity,
# eligibility accounting, and recruitment-source robustness
# ============================================================================

def report_agreement_descriptives(long_df):
    """
    Descriptives for plausibility and confidence change split by whether the
    participant's initial judgment agreed with the AI. Feeds the Results text
    directly (paper reports these means with SDs and cell sizes).
    """
    df = long_df.dropna(subset=['plaus', 'pre', 'ai']).copy()
    df['agreement'] = (df['pre'] == df['ai']).astype(int)

    print("\n" + "="*60)
    print("AGREEMENT-CONDITIONED DESCRIPTIVES")
    print("="*60)

    for label, val in [("Agreement (Human = AI)", 1), ("Disagreement (Human != AI)", 0)]:
        sub = df[df['agreement'] == val]
        print(f"\n{label}:  n = {len(sub)}")
        print(f"  Plausibility:      M = {sub['plaus'].mean():.3f}, SD = {sub['plaus'].std():.3f}")
        dc = sub['delta_conf'].dropna()
        print(f"  Confidence change: M = {dc.mean():.3f}, SD = {dc.std():.3f}")

    return df


def report_plausibility_diagnosticity(long_df):
    """
    Is perceived plausibility informative about whether the participant is
    actually right? Reports final accuracy at each plausibility level plus the
    overall correlation. A near-zero correlation is a substantive finding: it
    means plausibility drives reliance while carrying no signal about quality.
    """
    df = long_df.dropna(subset=['plaus', 'post', 'gt']).copy()
    df['post_correct'] = (df['post'] == df['gt']).astype(int)

    print("\n" + "="*60)
    print("PLAUSIBILITY DIAGNOSTICITY (plausibility vs. actual correctness)")
    print("="*60)

    for level in sorted(df['plaus'].dropna().unique()):
        sub = df[df['plaus'] == level]
        print(f"  Plausibility {int(level)}: Accuracy = {sub['post_correct'].mean():.3f}, "
              f"SD = {sub['post_correct'].std():.3f}, n = {len(sub)}")

    r, p = stats.pearsonr(df['plaus'], df['post_correct'])
    print(f"\n  Pearson r = {r:.4f}, p = {p:.4f}")
    print("  (A near-zero r means plausibility carries no information about decision quality.)")

    return {'r': r, 'p': p}


def report_eligibility_accounting(long_df):
    """
    Explain why RAIR models have fewer clusters than RSR models: some
    participants contribute no RAIR-eligible trials. The paper must state the
    reason correctly rather than assume it.
    """
    print("\n" + "="*60)
    print("ELIGIBILITY ACCOUNTING (why cluster counts differ across models)")
    print("="*60)

    all_participants = set(long_df['participant'].unique())

    rair_elig = long_df[(long_df['ai_correct']==1) & (long_df['human_pre_correct']==0)]
    rsr_elig = long_df[(long_df['ai_correct']==0) & (long_df['human_pre_correct']==1)]

    rair_participants = set(rair_elig['participant'].unique())
    rsr_participants = set(rsr_elig['participant'].unique())

    print(f"  Total participants:                  {len(all_participants)}")
    print(f"  With >=1 RAIR-eligible trial:        {len(rair_participants)}")
    print(f"  With >=1 RSR-eligible trial:         {len(rsr_participants)}")
    print(f"  RAIR-eligible trials (total):        {len(rair_elig)}")
    print(f"  RSR-eligible trials (total):         {len(rsr_elig)}")

    missing_rair = all_participants - rair_participants
    if missing_rair:
        print(f"\n  Participants with NO RAIR-eligible trials: {len(missing_rair)}")
        print("  Reason: a RAIR-eligible trial requires AI correct AND participant")
        print("  initially wrong. These participants were never initially wrong on a")
        print("  trial where the AI happened to be correct.")
        for pid in sorted(missing_rair):
            p_data = long_df[long_df['participant'] == pid]
            print(f"    participant {pid}: pre-accuracy = {p_data['human_pre_correct'].mean():.3f}, "
                  f"n_trials = {len(p_data)}")

    return {
        'n_total': len(all_participants),
        'n_rair_clusters': len(rair_participants),
        'n_rsr_clusters': len(rsr_participants)
    }


def run_recruitment_robustness(df_trials, long_df, recruitment_col=None,
                               prolific_label=PROLIFIC_LABEL):
    """
    Re-estimate the four significant hypotheses (H13-H16) separately by
    recruitment source, and test whether recruitment source moderates them.

    Addresses the Limitations point: roughly half the sample came from the
    authors' personal networks, so reviewers will ask whether the headline
    effects survive in the panel-recruited subsample alone.

    Set RECRUITMENT_COLUMN at the top of this file to enable.
    """
    print("\n" + "="*60)
    print("RECRUITMENT-SOURCE ROBUSTNESS CHECK")
    print("="*60)

    if recruitment_col is None:
        print("  SKIPPED: set RECRUITMENT_COLUMN at the top of this script to the")
        print("  column in your trials file that records recruitment source.")
        return None

    if recruitment_col not in df_trials.columns:
        print(f"  SKIPPED: column '{recruitment_col}' not found in the trials file.")
        print(f"  Available columns include: {list(df_trials.columns)[:10]} ...")
        return None

    source = df_trials[[recruitment_col]].copy()
    source['participant'] = source.index
    source.columns = ['recruitment', 'participant']
    df = long_df.merge(source, on='participant', how='left')
    df['is_prolific'] = (df['recruitment'].astype(str).str.strip().str.lower()
                         == str(prolific_label).strip().lower()).astype(int)

    n_prolific = df[df['is_prolific'] == 1]['participant'].nunique()
    n_other = df[df['is_prolific'] == 0]['participant'].nunique()
    print(f"  Prolific participants: {n_prolific}")
    print(f"  Other participants:    {n_other}")

    subgroup_specs = {
        'H13 (Plausibility -> RAIR)': (
            f"changed_to_correct ~ {LOO_PLAUSIBILITY_COLUMN}",
            lambda d: d[(d['ai_correct']==1) & (d['human_pre_correct']==0)],
            'logit'),
        'H14 (Plausibility -> RSR)': (
            f"stayed_correct ~ {LOO_PLAUSIBILITY_COLUMN}",
            lambda d: d[(d['ai_correct']==0) & (d['human_pre_correct']==1)],
            'logit'),
        'H15 (Plausibility -> dConf)': (
            f"delta_conf ~ {LOO_PLAUSIBILITY_COLUMN}",
            lambda d: d.dropna(subset=['delta_conf', LOO_PLAUSIBILITY_COLUMN]),
            'ols'),
    }

    rows = []
    for label, (formula, subset_fn, kind) in subgroup_specs.items():
        print(f"\n  {label}")
        for grp_name, grp_val in [("Prolific", 1), ("Personal network", 0)]:
            sub = subset_fn(df[df['is_prolific'] == grp_val].copy())
            if sub['participant'].nunique() < 5 or len(sub) < 20:
                print(f"    {grp_name:18} | insufficient data (n={len(sub)})")
                continue
            try:
                m = (logit_clustered(formula, sub) if kind == 'logit'
                     else ols_clustered(formula, sub))
                res = summarize(m)
                pred = get_main_predictor(res)
                coef = res['params'][pred]
                p_val = res['pvalues'][pred]
                lo, hi = get_ci(res, pred)
                print(f"    {grp_name:18} | beta={coef:7.4f}, 95% CI {format_ci(lo, hi)}, "
                      f"p={p_val:.4f} {sig_stars(p_val)}, N={res['n']}")
                rows.append({'Hypothesis': label, 'Group': grp_name, 'beta': coef,
                             'CI_lower': lo, 'CI_upper': hi, 'p_value': p_val,
                             'N': res['n']})
            except Exception as e:
                print(f"    {grp_name:18} | model failed: {e}")

        # Interaction test: does recruitment source moderate the effect?
        sub_all = subset_fn(df.copy())
        pred_var = formula.split('~')[1].strip()
        inter_formula = f"{formula} * is_prolific"
        try:
            m = (logit_clustered(inter_formula, sub_all) if kind == 'logit'
                 else ols_clustered(inter_formula, sub_all))
            res = summarize(m)
            inter_terms = [k for k in res['params'] if ':' in k]
            for term in inter_terms:
                p_val = res['pvalues'][term]
                lo, hi = get_ci(res, term)
                print(f"    interaction {term:22} | beta={res['params'][term]:7.4f}, "
                      f"95% CI {format_ci(lo, hi)}, p={p_val:.4f} {sig_stars(p_val)}")
                print("      (non-significant interaction => effect does not differ by source)")
        except Exception as e:
            print(f"    interaction model failed: {e}")

    if rows:
        out = pd.DataFrame(rows)
        try:
            out_path = Path(str(config.OUTPUT_CSV_FILES['hypothesis_summary'])).parent / "recruitment_robustness.csv"
            out.to_csv(str(out_path), index=False)
            print(f"\n  Saved to: {out_path}")
        except Exception as e:
            print(f"  Could not save CSV: {e}")
        return out
    return None

def compute_rair_rsr_by_age(df_trials, long_df):
    """
    Compute RAIR and RSR by age groups
    """
    # Merge age information into long_df
    age_col = 'What is your age?'
    if age_col not in df_trials.columns:
        print("Age column not found in data")
        return

    # Get age for each participant (using index as participant ID)
    participant_age = df_trials[[age_col]].copy()
    participant_age['participant'] = participant_age.index
    participant_age.columns = ['age_group', 'participant']

    # Merge with long_df
    long_with_age = long_df.merge(participant_age, on='participant', how='left')

    # Define age order
    age_order = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']

    print("\n" + "="*80)
    print("RAIR AND RSR BY AGE GROUP")
    print("="*80)

    results = []

    for age_group in age_order:
        age_data = long_with_age[long_with_age['age_group'] == age_group]

        if len(age_data) == 0:
            continue

        # RAIR: AI correct & human initially wrong
        rair_eligible = age_data[(age_data['ai_correct']==1) & (age_data['human_pre_correct']==0)]
        rair = rair_eligible['changed_to_correct'].mean() if len(rair_eligible) > 0 else float('nan')
        rair_n = len(rair_eligible)

        # RSR: AI wrong & human initially correct
        rsr_eligible = age_data[(age_data['ai_correct']==0) & (age_data['human_pre_correct']==1)]
        rsr = rsr_eligible['stayed_correct'].mean() if len(rsr_eligible) > 0 else float('nan')
        rsr_n = len(rsr_eligible)

        # AOR: average of RAIR and RSR
        aor = (rair + rsr) / 2.0 if not (pd.isna(rair) or pd.isna(rsr)) else float('nan')

        # Count unique participants in this age group
        n_participants = age_data['participant'].nunique()

        results.append({
            'Age Group': age_group,
            'N Participants': n_participants,
            'RAIR': rair,
            'RAIR_n': rair_n,
            'RSR': rsr,
            'RSR_n': rsr_n,
            'AOR': aor
        })

        print(f"\n{age_group}:")
        print(f"  Participants: {n_participants}")
        print(f"  RAIR: {rair:.3f} (n={rair_n} eligible trials)" if not pd.isna(rair) else f"  RAIR: N/A (n={rair_n})")
        print(f"  RSR:  {rsr:.3f} (n={rsr_n} eligible trials)" if not pd.isna(rsr) else f"  RSR:  N/A (n={rsr_n})")
        print(f"  AOR:  {aor:.3f}" if not pd.isna(aor) else f"  AOR:  N/A")

    # Create DataFrame for easier viewing
    results_df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("SUMMARY TABLE:")
    print("="*80)
    print(results_df.to_string(index=False))

    return results_df

def run_normality_tests(long_df):
    """
    Run Shapiro-Wilk normality tests on key continuous variables
    Similar to Schemmer et al.'s approach
    """
    normality_results = {}

    # Test continuous variables
    variables_to_test = ['delta_conf', INDIVIDUAL_PLAUSIBILITY_COLUMN]

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
    Create a summary table of hypothesis test results for easy interpretation.

    Includes 95% confidence intervals for every coefficient, odds ratios (with
    CIs) for logistic models, and a Bonferroni-corrected significance flag.
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
        "H14": "Plausibility → RSR",
        "H15": "Plausibility → Δ-Confidence",
        "H16": "Agreement (Human=AI) → Plausibility"
    }

    for h_key, res in results.items():
        if 'error' not in res:
            params = res.get('params', {})
            pvalues = res.get('pvalues', {})
            ses = res.get('bse', {})

            # Get the main predictor (not intercept)
            predictor = get_main_predictor(res)
            if predictor:
                coef = params[predictor]
                p_val = pvalues[predictor]
                se = ses.get(predictor, np.nan)
                ci_lower, ci_upper = get_ci(res, predictor)

                sig_level = sig_stars(p_val)
                supported = "Yes" if p_val < ALPHA else "No"
                survives_bonf = "Yes" if p_val < BONFERRONI_ALPHA else "No"

                row = {
                    'Hypothesis': h_key,
                    'Relationship': hypothesis_descriptions.get(h_key, h_key),
                    'Test': res.get('test_type', 'N/A'),
                    'β': coef,
                    'SE': se,
                    'CI_lower': ci_lower,
                    'CI_upper': ci_upper,
                    'CI_95': format_ci(ci_lower, ci_upper),
                    'p-value': p_val,
                    'Sig': sig_level,
                    'Supported': supported,
                    'Survives_Bonferroni': survives_bonf,
                    'N': res.get('n', 'N/A'),
                    'N_clusters': res.get('n_clusters', 'N/A')
                }

                # Odds ratios only make sense for logistic models
                if is_logistic(res):
                    row['OR'] = np.exp(coef)
                    row['OR_CI_lower'] = np.exp(ci_lower) if not pd.isna(ci_lower) else np.nan
                    row['OR_CI_upper'] = np.exp(ci_upper) if not pd.isna(ci_upper) else np.nan
                    row['OR_CI_95'] = format_ci(row['OR_CI_lower'], row['OR_CI_upper'])
                else:
                    row['OR'] = np.nan
                    row['OR_CI_lower'] = np.nan
                    row['OR_CI_upper'] = np.nan
                    row['OR_CI_95'] = ''

                summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def create_latex_hypothesis_table(summary_table):
    """
    Emit the hypothesis table as LaTeX rows, ready to paste into the paper.
    Saves manual re-typing of numbers (a recurring source of transcription
    errors between the analysis output and the manuscript).
    """
    lines = []
    lines.append("% Auto-generated from hypothesis_tests.py -- do not hand-edit numbers.")
    lines.append("% Columns: Hyp. & X -> Y & Model/Test & beta & 95% CI & p & N")
    for _, r in summary_table.iterrows():
        p_val = r['p-value']
        p_str = "$<.001$" if p_val < 0.001 else f"{p_val:.3f}"
        stars = ""
        if p_val < 0.001:
            stars = "$^{\\ast\\ast\\ast}$"
        elif p_val < 0.01:
            stars = "$^{\\ast\\ast}$"
        elif p_val < 0.05:
            stars = "$^{\\ast}$"

        beta_str = f"${r['β']:+.3f}$"
        ci_str = f"$[{r['CI_lower']:.3f}, {r['CI_upper']:.3f}]$"

        if p_val < ALPHA:
            beta_str = f"$\\mathbf{{{r['β']:+.3f}}}$"
            p_str = f"$\\mathbf{{{p_str.strip('$')}}}$"

        test_short = "Logit (z-test)" if "Logistic" in str(r['Test']) else "OLS (t-test)"
        lines.append(
            f"{r['Hypothesis']} & {r['Relationship']} & {test_short} & "
            f"{beta_str} & {ci_str} & {p_str}{stars} & {r['N']} \\\\"
        )
    return "\n".join(lines)

# Model specification for every hypothesis, used by the two-way clustered
# robustness pass below. Kept in one place so it cannot drift from test_H1..H16.
TWOWAY_SPECS = [
    ("H1",  "changed_to_correct ~ faith",       "rair",  "logit"),
    ("H2",  "stayed_correct ~ faith",           "rsr",   "logit"),
    ("H3",  "delta_conf ~ faith",               "all",   "ols"),
    ("H4",  "post_correct ~ faith",             "post",  "logit"),
    ("H5",  "plaus ~ faith",                    "all",   "ols"),
    ("H6",  "changed_to_correct ~ delta_conf",  "rair",  "logit"),
    ("H7",  "stayed_correct ~ delta_conf",      "rsr",   "logit"),
    ("H8",  "plaus ~ model_size",               "all",   "ols"),
    ("H9",  "changed_to_correct ~ model_size",  "rair",  "logit"),
    ("H10", "stayed_correct ~ model_size",      "rsr",   "logit"),
    ("H11", "delta_conf ~ model_size",          "all",   "ols"),
    ("H12", "post_correct ~ model_size",        "post",  "logit"),
    ("H13", "changed_to_correct ~ plaus_loo",   "rair",  "logit"),
    ("H14", "stayed_correct ~ plaus_loo",       "rsr",   "logit"),
    ("H15", "delta_conf ~ plaus_loo",           "all",   "ols"),
    ("H16", "plaus ~ agreement",                "agree", "ols"),
]


def _twoway_subset(long_df, kind):
    if kind == "rair":
        return long_df[(long_df["ai_correct"] == 1) & (long_df["human_pre_correct"] == 0)].copy()
    if kind == "rsr":
        return long_df[(long_df["ai_correct"] == 0) & (long_df["human_pre_correct"] == 1)].copy()
    if kind == "post":
        d = long_df.copy()
        d["post_correct"] = (d["post"] == d["gt"]).astype(int)
        return d
    if kind == "agree":
        d = long_df.dropna(subset=[INDIVIDUAL_PLAUSIBILITY_COLUMN, "pre", "ai"]).copy()
        d["agreement"] = (d["pre"] == d["ai"]).astype(int)
        return d
    return long_df.copy()


def run_twoway_robustness(long_df):
    """
    Re-fit all sixteen hypotheses clustering on participant AND question.

    Why: `faith` is fixed per item (odd Q = faithful), and plaus_loo takes
    essentially 16 distinct values. Clustering on participant alone treats those
    16 explanations as if they were 1632 independent observations, which
    understates the uncertainty of any item-level contrast. The point estimates are
    identical to the main table by construction -- only the SEs and p-values move.
    """
    rows = []
    for name, formula, subset_kind, kind in TWOWAY_SPECS:
        df = _twoway_subset(long_df, subset_kind)
        predictor = formula.split("~")[1].strip()
        df = df.dropna(subset=[formula.split("~")[0].strip(), predictor, "participant", "Q"])
        try:
            fit = (logit_clustered if kind == "logit" else ols_clustered)(
                formula, df, cluster_var=["participant", "Q"])
            ci = fit.conf_int().loc[predictor]
            rows.append({
                "Hypothesis": name,
                "Model": formula,
                "beta": fit.params[predictor],
                "SE_2way": fit.bse[predictor],
                "p_2way": fit.pvalues[predictor],
                "CI_95_2way": format_ci(ci[0], ci[1]),
                "Sig_2way": sig_stars(fit.pvalues[predictor]),
                "Survives_Bonferroni_2way": bool(fit.pvalues[predictor] < BONFERRONI_ALPHA),
                "N": int(fit.nobs),
            })
        except Exception as e:
            rows.append({"Hypothesis": name, "Model": formula, "beta": np.nan,
                         "SE_2way": np.nan, "p_2way": np.nan, "CI_95_2way": "",
                         "Sig_2way": f"failed: {e}", "Survives_Bonferroni_2way": False,
                         "N": 0})
    return pd.DataFrame(rows)


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
    print(f"Note: {N_HYPOTHESES} tests -> Bonferroni alpha = {BONFERRONI_ALPHA:.5f}")
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
        "H15": test_H15(long_df, normality_results),
        "H16": test_H16(long_df, normality_results),
    }
    # Two-way (participant x question) clustered robustness pass.
    print("\n" + "=" * 60)
    print("ROBUSTNESS: TWO-WAY CLUSTERED SEs (participant x question)")
    print("=" * 60)
    print("Same coefficients as above; only the SEs and p-values change.")
    twoway = run_twoway_robustness(long_df)
    print(twoway[["Hypothesis", "beta", "SE_2way", "CI_95_2way", "p_2way",
                  "Sig_2way", "Survives_Bonferroni_2way", "N"]].to_string(index=False))
    twoway_path = Path(str(config.OUTPUT_CSV_FILES["hypothesis_summary"])).parent / "hypothesis_twoway_clustered.csv"
    twoway.to_csv(str(twoway_path), index=False)
    print(f"\n\u2713 Two-way clustered table saved to: {twoway_path}")

    return results, long_df, normality_results

def main():
    df_trials = pd.read_excel(str(config.PROCESSED_DATA_FILE))

    results, long_df, normality_results = run_all_hypotheses(df_trials, n_trials=config.N_TRIALS)

    # Create and display summary table
    print("\n" + "="*60)
    print("HYPOTHESIS SUMMARY TABLE")
    print("="*60)
    summary_table = create_hypothesis_summary_table(results)

    # Console view: the columns needed to write up Results
    display_cols = ['Hypothesis', 'Relationship', 'β', 'CI_95', 'p-value',
                    'Sig', 'Supported', 'Survives_Bonferroni', 'N']
    print(summary_table[display_cols].to_string(index=False))

    # Odds-ratio view for the logistic models
    logit_rows = summary_table[summary_table['OR'].notna()]
    if len(logit_rows) > 0:
        print("\n" + "-"*60)
        print("ODDS RATIOS (logistic models only)")
        print("-"*60)
        print(logit_rows[['Hypothesis', 'Relationship', 'OR', 'OR_CI_95',
                          'p-value', 'Sig']].to_string(index=False))

    # Save summary table to CSV
    summary_table.to_csv(str(config.OUTPUT_CSV_FILES["hypothesis_summary"]), index=False)
    print(f"\n✓ Summary table saved to: {config.OUTPUT_CSV_FILES['hypothesis_summary']}")

    # Emit LaTeX rows for the manuscript table
    try:
        latex_rows = create_latex_hypothesis_table(summary_table)
        latex_path = Path(str(config.OUTPUT_CSV_FILES['hypothesis_summary'])).parent / "hypothesis_table.tex"
        with open(str(latex_path), 'w') as f:
            f.write(latex_rows)
        print(f"✓ LaTeX table rows saved to: {latex_path}")
        print("\n" + "-"*60)
        print("LATEX TABLE ROWS (paste into the manuscript)")
        print("-"*60)
        print(latex_rows)
    except Exception as e:
        print(f"Could not generate LaTeX table: {e}")

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
            ses = res.get("bse", {})

            print(f"  Coefficients (with 95% CI):")
            for param_name, coef in params.items():
                p_val = pvalues.get(param_name, np.nan)
                se = ses.get(param_name, np.nan)
                lo, hi = get_ci(res, param_name)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                print(f"    {param_name:20} β={coef:8.4f}, SE={se:6.4f}, "
                      f"95% CI {format_ci(lo, hi)}, p={p_val:.4f} {sig}")
                # Odds ratio for logistic models
                if is_logistic(res) and param_name.lower() != 'intercept':
                    print(f"    {'':20} OR={np.exp(coef):8.4f}, "
                          f"95% CI {format_ci(np.exp(lo), np.exp(hi))}")

            # Bonferroni flag on the main predictor
            pred = get_main_predictor(res)
            if pred:
                p_main = pvalues.get(pred, np.nan)
                bonf = "SURVIVES" if p_main < BONFERRONI_ALPHA else "does not survive"
                print(f"  Bonferroni (alpha={BONFERRONI_ALPHA:.5f}): {bonf}")

            print(f"  Number of observations: {res.get('n', 'N/A')}")
            if 'n_clusters' in res:
                print(f"  Number of clusters (participants): {res.get('n_clusters', 'N/A')}")

            # Show normality info if available
            if 'normality' in res:
                norm_info = res['normality']
                if norm_info.get('normal') is not None:
                    status = "NORMAL" if norm_info['normal'] else "NOT NORMAL"
                    print(f"  Normality: {status} (W={norm_info['statistic']:.4f}, p={norm_info['p_value']:.4f})")

    # Why cluster counts differ between RAIR and RSR models
    report_eligibility_accounting(long_df)

    # Agreement-conditioned descriptives (plausibility + confidence change)
    report_agreement_descriptives(long_df)

    # Is plausibility diagnostic of actual correctness?
    report_plausibility_diagnosticity(long_df)

    # Recruitment-source robustness (enable via RECRUITMENT_COLUMN)
    run_recruitment_robustness(df_trials, long_df, recruitment_col=RECRUITMENT_COLUMN)

    # RAIR and RSR by age groups
    compute_rair_rsr_by_age(df_trials, long_df)

    print("\n" + "="*60)
    print("DESCRIPTIVE STATISTICS BY GROUP")
    print("="*60)

    # Overall confidence: pre, post, and delta (reported in the paper's Overview)
    print("\n--- Overall Confidence (pre / post / delta) ---")
    dc = long_df['delta_conf'].dropna()
    print(f"  Delta confidence: M = {dc.mean():.4f}, SD = {dc.std():.4f}, n = {len(dc)}")
    print("  NOTE: pre/post confidence means are computed in "
          "plot_confidence_plausibility_distribution(); the paper reports those "
          "values, so make sure the delta above equals post - pre.")

    # Overall plausibility (reported in the paper's Overview)
    pl = long_df['plaus'].dropna()
    print(f"\n--- Overall Plausibility ---")
    print(f"  M = {pl.mean():.4f}, SD = {pl.std():.4f}, n = {len(pl)}")

    # Overall RAIR / RSR pooled across trials (the values reported in the paper)
    rair_elig = long_df[(long_df['ai_correct']==1) & (long_df['human_pre_correct']==0)]
    rsr_elig = long_df[(long_df['ai_correct']==0) & (long_df['human_pre_correct']==1)]
    print(f"\n--- Pooled Trial-Level RAIR / RSR ---")
    print(f"  RAIR = {rair_elig['changed_to_correct'].mean():.4f} "
          f"({int(rair_elig['changed_to_correct'].sum())}/{len(rair_elig)} eligible trials)")
    print(f"  RSR  = {rsr_elig['stayed_correct'].mean():.4f} "
          f"({int(rsr_elig['stayed_correct'].sum())}/{len(rsr_elig)} eligible trials)")
    print("  NOTE: these pooled values differ from the participant-level means")
    print("  below; the paper reports the pooled trial-level figures.")

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

    print("\n--- Mean Plausibility by Faithfulness ---")
    print(descriptive_stats_by_group(long_df, INDIVIDUAL_PLAUSIBILITY_COLUMN, 'faith'))

    print("\n--- Delta Confidence by Model Size ---")
    print(descriptive_stats_by_group(long_df, 'delta_conf', 'model_size'))

    print("\n--- Mean Plausibility by Model Size ---")
    print(descriptive_stats_by_group(long_df, INDIVIDUAL_PLAUSIBILITY_COLUMN, 'model_size'))

    print("\n=== Long DataFrame Summary ===")
    print(f"Total observations: {len(long_df)}")
    print(f"\nDataFrame Description:")
    print(long_df.describe())

    # Participant-level aggregates (Schemmer-style between-subjects analysis)
    print("\n" + "="*60)
    print("PARTICIPANT-LEVEL AGGREGATES")
    print("="*60)

    participant_df = compute_participant_aggregates(long_df)
    n_unique_participants = participant_df['participant'].nunique()
    print(f"\nRows in participant aggregate table: {len(participant_df)}")
    print(f"Unique participants: {n_unique_participants}")
    print("(Each participant contributes up to 4 rows: 2 faith conditions + 2 model sizes.)")
    print(f"\nParticipant-level summary:")
    print(participant_df.describe())

    # Save participant aggregates
    participant_df.to_csv(str(config.OUTPUT_CSV_FILES["participant_aggregates"]), index=False)
    print(f"\n✓ Participant aggregates saved to: {config.OUTPUT_CSV_FILES['participant_aggregates']}")

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
            for g, stats_d in result['groups'].items():
                label = "Faithful" if g == 1 else "Unfaithful"
                print(f"  {label:12} | M={stats_d['mean']:.4f}, SD={stats_d['std']:.4f}")
            sig = sig_stars(result['p_value'])
            print(f"  Statistic={result['statistic']:.4f}, p={result['p_value']:.4f} {sig}")
            lo, hi = result.get('difference_ci', (np.nan, np.nan))
            print(f"  Mean Difference={result['difference']:.4f}, 95% CI {format_ci(lo, hi)}")
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
                'Diff_CI_lower': lo,
                'Diff_CI_upper': hi,
                'Statistic': result['statistic'],
                'p_value': result['p_value'],
                'Effect_Size': result['effect_size'],
                'Effect_Size_Type': result['effect_size_name'],
                'Significant': 'Yes' if result['p_value'] < ALPHA else 'No'
            })

    # Export within-subjects comparisons
    if faith_comparisons:
        faith_df_export = pd.DataFrame(faith_comparisons)
        faith_df_export.to_csv(str(config.OUTPUT_CSV_FILES["within_subjects_faithfulness"]), index=False)
        print(f"\n✓ Faithfulness comparisons saved to: {config.OUTPUT_CSV_FILES['within_subjects_faithfulness']}")

    print("\n\n--- Comparisons by MODEL SIZE (Paired Tests) ---")
    model_size_comparisons = []

    # Filter to only model_size-related rows
    size_df = participant_df[participant_df['model_size'].notna()].copy()

    for var in variables_to_test:
        result = within_subjects_test(size_df, var, 'model_size')
        if 'error' in result:
            print(f"\n{var}: {result['error']}")
            if var in ('human_pre_accuracy', 'post_accuracy'):
                print("  (Expected: compute_participant_aggregates() does not record these")
                print("   two fields in the model_size rows, only in the faith rows.)")
        else:
            print(f"\n{var}:")
            print(f"  Test: {result['test']} (N pairs = {result['n_paired']})")
            for g, stats_d in result['groups'].items():
                label = "Large LLM" if g == 1 else "Small LLM"
                print(f"  {label:12} | M={stats_d['mean']:.4f}, SD={stats_d['std']:.4f}")
            sig = sig_stars(result['p_value'])
            print(f"  Statistic={result['statistic']:.4f}, p={result['p_value']:.4f} {sig}")
            lo, hi = result.get('difference_ci', (np.nan, np.nan))
            print(f"  Mean Difference={result['difference']:.4f}, 95% CI {format_ci(lo, hi)}")
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
                'Diff_CI_lower': lo,
                'Diff_CI_upper': hi,
                'Statistic': result['statistic'],
                'p_value': result['p_value'],
                'Effect_Size': result['effect_size'],
                'Effect_Size_Type': result['effect_size_name'],
                'Significant': 'Yes' if result['p_value'] < ALPHA else 'No'
            })

    # Export within-subjects comparisons
    if model_size_comparisons:
        model_size_df_export = pd.DataFrame(model_size_comparisons)
        model_size_df_export.to_csv(str(config.OUTPUT_CSV_FILES["within_subjects_modelsize"]), index=False)
        print(f"\n✓ Model size comparisons saved to: {config.OUTPUT_CSV_FILES['within_subjects_modelsize']}")

    # Plots
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    general_plots_dir = config.PLOT_DIRS["general"]
    plot_mean_rair_rsr_by_faith(long_df, out_path=str(general_plots_dir / "mean_rair_rsr_by_faith.png"))
    plot_mean_conf_change_by_faith(long_df, out_path=str(general_plots_dir / "mean_conf_change_by_faith.png"))
    plot_mean_final_accuracy_by_faith(long_df, out_path=str(general_plots_dir / "mean_final_accuracy_by_faith.png"))
    plot_plausibility_violin_by_faith(long_df, out_path=str(general_plots_dir / "plausibility_violin_by_faith.png"))
    plot_per_question_accuracy(long_df, out_path=str(general_plots_dir / "per_question_accuracy.png"))
    plot_per_question_accuracy_by_modelsize(long_df, out_path=str(general_plots_dir / "per_question_accuracy_by_modelsize.png"))
    plot_per_question_accuracy_by_faithfulness(long_df, out_path=str(general_plots_dir / "per_question_accuracy_by_faithfulness.png"))
    plot_rair_rsr_per_question(long_df, out_path=str(general_plots_dir / "rair_rsr_per_question.png"))
    plot_human_accuracy_before_after(long_df, out_path=str(general_plots_dir / "human_accuracy_before_after.png"))
    plot_confidence_plausibility_distribution(df_trials, out_path=str(general_plots_dir / "confidence_plausibility_distribution.png"))
    plot_aor_scatter_by_faith(long_df, out_path=str(general_plots_dir / "aor_by_faith_scatter.png"))
    plot_aor_scatter_by_modelsize(long_df, out_path=str(general_plots_dir / "aor_by_modelsize_scatter.png"))
    plot_plausibility_vs_accuracy(long_df, out_path=str(general_plots_dir / "plausibility_vs_accuracy.png"))
    plot_plausibility_vs_conf_change(long_df, out_path=str(general_plots_dir / "plausibility_vs_conf_change.png"))
    plot_plausibility_by_agreement(long_df, out_path=str(general_plots_dir / "plausibility_by_agreement.png"))
    plot_conf_change_by_agreement(long_df, out_path=str(general_plots_dir / "conf_change_by_agreement.png"))
    plot_conf_vs_rair_scatter(long_df, out_path=str(general_plots_dir / "conf_vs_rair_scatter.png"))
    plot_conf_vs_rsr_scatter(long_df, out_path=str(general_plots_dir / "conf_vs_rsr_scatter.png"))
    plot_rair_rsr_by_modelsize(long_df, out_path=str(general_plots_dir / "rair_rsr_by_modelsize.png"))
    plot_conf_change_by_modelsize(long_df, out_path=str(general_plots_dir / "conf_change_by_modelsize.png"))
    plot_accuracy_by_modelsize(long_df, out_path=str(general_plots_dir / "accuracy_by_modelsize.png"))
    plot_plaus_vs_rair_rsr(long_df, out_path=str(general_plots_dir / "plaus_vs_rair_rsr.png"))
    plot_plausibility_by_modelsize(long_df, out_path=str(general_plots_dir / "plausibility_by_modelsize.png"))
    print("✓ All visualizations generated")

    # Final summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE - OUTPUT FILES")
    print("="*60)
    print("\nCSV Files Generated:")
    print(f"  • {config.OUTPUT_CSV_FILES['hypothesis_summary']}")
    print(f"  • {config.OUTPUT_CSV_FILES['participant_aggregates']}")
    print(f"  • {config.OUTPUT_CSV_FILES['within_subjects_faithfulness']}")
    print(f"  • {config.OUTPUT_CSV_FILES['within_subjects_modelsize']}")
    print("\nAll visualization plots have been saved as PNG files.")
    print("\nNote: This analysis uses WITHIN-SUBJECTS (repeated measures) tests")
    print("because each participant experienced both conditions (faithful/unfaithful).")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()