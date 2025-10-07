import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from plots import (
    plot_mean_rair_rsr_by_faith,
    plot_mean_conf_change_by_faith,
    plot_mean_final_accuracy_by_faith,
    plot_plausibility_violin_by_faith,
    plot_per_question_accuracy,
    plot_conf_vs_rair_scatter,
    plot_conf_vs_rsr_scatter,
    plot_rair_rsr_by_modelsize,
    plot_conf_change_by_modelsize,
    plot_plaus_vs_rair_rsr,
    plot_accuracy_by_modelsize,
    plot_plausibility_by_modelsize,
    plot_confidence_plausibility_distribution
)

def logit(formula, data):
    model = smf.logit(formula, data=data.dropna()).fit(disp=False)
    return model

def ols(formula, data):
    model = smf.ols(formula, data=data.dropna()).fit()
    return model

def summarize(model):
    return {
        "params": model.params.to_dict(),
        "pvalues": model.pvalues.to_dict(),
        "conf_int": model.conf_int().to_dict(),
        "n": int(model.nobs)
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
def test_H1(long_df):
    df = long_df[(long_df["ai_correct"]==1) & (long_df["human_pre_correct"]==0)].copy()
    # DV: changed_to_correct (binary), IV: faith
    m = logit("changed_to_correct ~ faith", df)
    return summarize(m)

# H2: Faithfulness -> RSR (among AI-wrong & human initially correct)
def test_H2(long_df):
    df = long_df[(long_df["ai_correct"]==0) & (long_df["human_pre_correct"]==1)].copy()
    m = logit("stayed_correct ~ faith", df)
    return summarize(m)

# H3: Faithfulness -> confidence calibration (participants' confidence better aligns with correctness)
def test_H3(long_df):
    df = long_df.copy()
    # Using confidence change as proxy for calibration
    m = ols("delta_conf ~ faith", df)
    return summarize(m)

# H4: Faithfulness affects final task accuracy (complementary team performance)
def test_H4(long_df):
    df = long_df.copy()
    df["post_correct"] = (df["post"] == df["gt"]).astype(int)
    m = logit("post_correct ~ faith", df)
    return summarize(m)

# H5: Faithfulness increases perceived plausibility
def test_H5(long_df):
    df = long_df.copy()
    m = ols("plaus ~ faith", df)
    return summarize(m)

# H6: Larger confidence changes predict higher RAIR (participants more often switch from wrong to correct)
def test_H6(long_df):
    df = long_df[(long_df["ai_correct"]==1) & (long_df["human_pre_correct"]==0)].copy()
    m = logit("changed_to_correct ~ delta_conf", df)
    return summarize(m)

# H7: Smaller or negative confidence changes predict higher RSR (participants resist incorrect AI advice)
def test_H7(long_df):
    df = long_df[(long_df["ai_correct"]==0) & (long_df["human_pre_correct"]==1)].copy()
    m = logit("stayed_correct ~ delta_conf", df)
    return summarize(m)

# H8: Larger LLMs produce more plausible explanations  (model_size: 1=large, 0=small)
def test_H8(long_df):
    m = ols("plaus ~ model_size", long_df)
    return summarize(m)

# H9: Larger LLMs produce higher RAIR (on RAIR-eligible subset)
def test_H9(long_df):
    df = long_df[(long_df["ai_correct"]==1) & (long_df["human_pre_correct"]==0)].copy()
    m = logit("changed_to_correct ~ model_size", df)
    return summarize(m)

# H10: Larger LLMs produce higher RSR (on RSR-eligible subset)
def test_H10(long_df):
    df = long_df[(long_df["ai_correct"]==0) & (long_df["human_pre_correct"]==1)].copy()
    m = logit("stayed_correct ~ model_size", df)
    return summarize(m)

# H11: Larger LLMs produce bigger confidence changes
def test_H11(long_df):
    m = ols("delta_conf ~ model_size", long_df)
    return summarize(m)

# H12: Larger LLMs lead to higher final task accuracy (complementary team performance)
def test_H12(long_df):
    df = long_df.copy()
    df["post_correct"] = (df["post"] == df["gt"]).astype(int)
    m = logit("post_correct ~ model_size", df)
    return summarize(m)

# H13: Higher perceived plausibility is associated with higher RAIR
def test_H13(long_df):
    df = long_df[(long_df["ai_correct"]==1) & (long_df["human_pre_correct"]==0)].copy()
    m = logit("changed_to_correct ~ plaus", df)
    return summarize(m)

# H14: Higher perceived plausibility is associated with higher RSR
def test_H14(long_df):
    df = long_df[(long_df["ai_correct"]==0) & (long_df["human_pre_correct"]==1)].copy()
    m = logit("stayed_correct ~ plaus", df)
    return summarize(m)

def run_all_hypotheses(df_trials, n_trials=16):
    long_df = make_long(df_trials, n_trials=n_trials)
    results=()

    results = {
        "H1":  test_H1(long_df),
        "H2":  test_H2(long_df),
        "H3":  test_H3(long_df),
        "H4":  test_H4(long_df),
        "H5":  test_H5(long_df),
        "H6":  test_H6(long_df),
        "H7":  test_H7(long_df),
        "H8":  test_H8(long_df),
        "H9":  test_H9(long_df),
        "H10": test_H10(long_df),
        "H11": test_H11(long_df),
        "H12": test_H12(long_df),
        "H13": test_H13(long_df),
        "H14": test_H14(long_df),
    }
    return results, long_df

def main():
    df_trials = pd.read_excel("experiment_results_with_metrics.xlsx")

    results, long_df = run_all_hypotheses(df_trials, n_trials=16)

    print("=== Hypothesis Test Results Summary ===")
    for key, res in results.items():
        print(f"{key}:")
        if isinstance(res, dict) and "error" in res:
            print(f"  Error: {res['error']}")
        else:
            params = res.get("params", {})
            pvalues = res.get("pvalues", {})
            print(f"  Params: {params}")
            print(f"  P-values: {pvalues}")
            print(f"  Number of observations: {res.get('n', 'N/A')}")
            print()
    print("\n=== Long DataFrame ===")
    print(long_df)
    print("\nDataFrame Info:")
    print(long_df.info())
    print("\nDataFrame Description:")
    print(long_df.describe())

    # Plots
    plot_mean_rair_rsr_by_faith(long_df)
    plot_mean_conf_change_by_faith(long_df)
    plot_mean_final_accuracy_by_faith(long_df)
    plot_plausibility_violin_by_faith(long_df)
    plot_per_question_accuracy(long_df)
    plot_confidence_plausibility_distribution(df_trials)
    plot_conf_vs_rair_scatter(long_df)
    plot_conf_vs_rsr_scatter(long_df)
    plot_rair_rsr_by_modelsize(long_df)
    plot_conf_change_by_modelsize(long_df)
    plot_accuracy_by_modelsize(long_df)
    plot_plaus_vs_rair_rsr(long_df)
    plot_plausibility_by_modelsize(long_df)

if __name__ == "__main__":
    main()
