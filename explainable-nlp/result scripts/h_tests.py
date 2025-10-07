import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
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

def plot_mean_rair_rsr_by_faith(long_df: pd.DataFrame, out_path: str = "mean_rair_rsr_by_faith.png") -> str:
    """
    Compute mean RAIR (on RAIR-eligible subset) and mean RSR (on RSR-eligible subset)
    grouped by faithfulness (1=faithful, 0=unfaithful), and save a side-by-side bar plot.
    """
    # RAIR-eligible: AI correct & human initially wrong
    df_rair = long_df[(long_df["ai_correct"]==1) & (long_df["human_pre_correct"]==0)].copy()
    df_rair["rair"] = df_rair["changed_to_correct"].astype(float)
    mean_rair = df_rair.groupby("faith")["rair"].mean().rename("mean_rair")

    # RSR-eligible: AI wrong & human initially correct
    df_rsr = long_df[(long_df["ai_correct"]==0) & (long_df["human_pre_correct"]==1)].copy()
    df_rsr["rsr"] = df_rsr["stayed_correct"].astype(float)
    mean_rsr = df_rsr.groupby("faith")["rsr"].mean().rename("mean_rsr")

    summary = pd.concat([mean_rair, mean_rsr], axis=1)
    summary = summary.reset_index()  # columns: faith, mean_rair, mean_rsr
    summary["faith_label"] = summary["faith"].map({1: "Faithful", 0: "Unfaithful"})

    # Plot
    plt.figure(figsize=(8,5))
    width = 0.35
    x = np.arange(len(summary))
    plt.bar(x - width/2, summary["mean_rair"], width=width, label="RAIR")
    plt.bar(x + width/2, summary["mean_rsr"],  width=width, label="RSR")
    plt.xticks(x, summary["faith_label"]) 
    plt.ylabel("Mean (proportion)")
    plt.title("Mean RAIR and RSR by Faithfulness")
    plt.ylim(0,1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot to {out_path}")
    return out_path

def plot_mean_conf_change_by_faith(long_df: pd.DataFrame, out_path: str = "mean_conf_change_by_faith.png") -> str:
    """
    Plot average confidence change (post - pre) by faithfulness (1=faithful, 0=unfaithful).
    Uses the per-trial 'delta_conf' field from long_df.
    """
    df = long_df.dropna(subset=["delta_conf"]).copy()
    summary = df.groupby("faith")["delta_conf"].mean().reset_index()
    summary["faith_label"] = summary["faith"].map({1: "Faithful", 0: "Unfaithful"})

    plt.figure(figsize=(6,4))
    sns.barplot(data=summary, x="faith_label", y="delta_conf", color="#4C78A8")
    plt.ylabel("Mean Confidence Change (post - pre)")
    plt.xlabel("")
    plt.title("Average Confidence Change by Faithfulness")
    plt.axhline(0, color="gray", linewidth=1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot to {out_path}")
    return out_path

def plot_mean_final_accuracy_by_faith(long_df: pd.DataFrame, out_path: str = "mean_final_accuracy_by_faith.png") -> str:
    """
    Plot mean final task accuracy (post == gt) under faithful vs unfaithful explanations,
    averaged across model sizes (i.e., ignore model_size in grouping).
    """
    df = long_df.copy()
    df["post_correct"] = (df["post"] == df["gt"]).astype(float)
    summary = df.groupby("faith")["post_correct"].mean().reset_index()
    summary["faith_label"] = summary["faith"].map({1: "Faithful", 0: "Unfaithful"})

    plt.figure(figsize=(6,4))
    sns.barplot(data=summary, x="faith_label", y="post_correct", color="#72B7B2")
    plt.ylabel("Mean Final Accuracy (post == GT)")
    plt.xlabel("")
    plt.title("Mean Final Task Accuracy by Faithfulness")
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot to {out_path}")
    return out_path

def plot_plausibility_violin_by_faith(long_df: pd.DataFrame, out_path: str = "plausibility_violin_by_faith.png") -> str:
    """
    Violin plot of plausibility (1–5 Likert) for faithful vs unfaithful.
    """
    df = long_df.dropna(subset=["plaus"]).copy()
    df["faith_label"] = df["faith"].map({1: "Faithful", 0: "Unfaithful"})

    plt.figure(figsize=(7,4))
    sns.violinplot(data=df, x="faith_label", y="plaus", inner="box", cut=0, palette=["#4C78A8", "#F58518"])
    plt.xlabel("")
    plt.ylabel("Plausibility (1–5)")
    plt.title("Plausibility by Faithfulness")
    plt.ylim(1,5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot to {out_path}")
    return out_path

def plot_per_question_accuracy(long_df: pd.DataFrame, out_path: str = "per_question_accuracy.png") -> str:
    """
    Plot accuracy per question (before and after AI explanations) across all participants.
    Shows proportion of correct responses for each of the 16 questions.
    """
    # Compute accuracy per question before (pre) and after (post)
    df = long_df.copy()
    df["pre_correct"] = (df["pre"] == df["gt"]).astype(float)
    df["post_correct"] = (df["post"] == df["gt"]).astype(float)
    
    # Group by question and compute mean accuracy
    accuracy_by_q = df.groupby("Q")[["pre_correct", "post_correct"]].mean().reset_index()
    accuracy_by_q.columns = ["Question", "Before", "After"]
    
    # Melt for easier plotting
    accuracy_long = pd.melt(accuracy_by_q, id_vars=["Question"], 
                             value_vars=["Before", "After"],
                             var_name="Timing", value_name="Accuracy")
    
    plt.figure(figsize=(12, 5))
    x = np.arange(1, 17)
    width = 0.35
    
    before = accuracy_by_q["Before"].values
    after = accuracy_by_q["After"].values
    
    plt.bar(x - width/2, before, width, label="Before (Initial)", color="#4C78A8", alpha=0.8)
    plt.bar(x + width/2, after, width, label="After (Post-explanation)", color="#F58518", alpha=0.8)
    
    # Add chance line at 0.5
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1.5, label="Chance (50%)")
    
    plt.xlabel("Question Number")
    plt.ylabel("Accuracy (Proportion Correct)")
    plt.title("Accuracy per Question Across All Participants")
    plt.xticks(x)
    plt.ylim(0, 1)
    plt.legend(loc="best")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot to {out_path}")
    return out_path

def plot_confidence_plausibility_distribution(df_trials: pd.DataFrame, out_path: str = "confidence_plausibility_distribution.png") -> str:
    """
    Plot distributions of confidence ratings (before/after, assumed 1-7 scale) and plausibility (1-5 scale).
    Uses wide format df_trials with columns like Q1_Conf1, Q1_Conf2, Q1_Plausibility for each question.
    """
    n_trials = 16
    
    # Collect confidence before (Conf1) and after (Conf2), and plausibility
    conf1_values = []
    conf2_values = []
    plaus_values = []
    
    for i in range(1, n_trials + 1):
        c1_col = f"Q{i}_Conf1"
        c2_col = f"Q{i}_Conf2"
        plaus_col = f"Q{i}_Plausibility"
        
        if c1_col in df_trials.columns:
            vals = pd.to_numeric(df_trials[c1_col], errors="coerce").dropna()
            conf1_values.extend(vals.tolist())
        
        if c2_col in df_trials.columns:
            vals = pd.to_numeric(df_trials[c2_col], errors="coerce").dropna()
            conf2_values.extend(vals.tolist())
        
        if plaus_col in df_trials.columns:
            vals = pd.to_numeric(df_trials[plaus_col], errors="coerce").dropna()
            plaus_values.extend(vals.tolist())
    
    # Create 2-panel figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left panel: Confidence distributions (before and after)
    ax1 = axes[0]
    if conf1_values and conf2_values:
        conf1_counts = pd.Series(conf1_values).value_counts().sort_index()
        conf2_counts = pd.Series(conf2_values).value_counts().sort_index()
        
        # Align indices (1-7 scale assumed)
        all_vals = list(range(1, 8))
        conf1_freq = [conf1_counts.get(v, 0) for v in all_vals]
        conf2_freq = [conf2_counts.get(v, 0) for v in all_vals]
        
        x = np.array(all_vals)
        width = 0.35
        ax1.bar(x - width/2, conf1_freq, width, label="Before (Conf1)", color="#4C78A8", alpha=0.8, edgecolor="black")
        ax1.bar(x + width/2, conf2_freq, width, label="After (Conf2)", color="#F58518", alpha=0.8, edgecolor="black")
        
        ax1.set_xlabel("Confidence Rating (1-7)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Distribution of Confidence Ratings")
        ax1.set_xticks(all_vals)
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)
        
        # Compute mean and SD
        conf1_mean = np.mean(conf1_values)
        conf1_sd = np.std(conf1_values, ddof=1)
        conf2_mean = np.mean(conf2_values)
        conf2_sd = np.std(conf2_values, ddof=1)
        
        ax1.text(0.02, 0.98, f"Before: M = {conf1_mean:.2f}, SD = {conf1_sd:.2f}\nAfter: M = {conf2_mean:.2f}, SD = {conf2_sd:.2f}", 
                 transform=ax1.transAxes, ha="left", va="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.9), fontsize=9)
        
        print(f"Confidence Before: M = {conf1_mean:.2f}, SD = {conf1_sd:.2f}")
        print(f"Confidence After: M = {conf2_mean:.2f}, SD = {conf2_sd:.2f}")
    else:
        ax1.text(0.5, 0.5, "Confidence data not available", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("Distribution of Confidence Ratings")
    
    # Right panel: Plausibility distribution
    ax2 = axes[1]
    if plaus_values:
        plaus_counts = pd.Series(plaus_values).value_counts().sort_index()
        ax2.bar(plaus_counts.index, plaus_counts.values, color="#72B7B2", alpha=0.8, edgecolor="black")
        ax2.set_xlabel("Plausibility Rating (1-5)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Distribution of Plausibility Ratings")
        ax2.set_xticks([1, 2, 3, 4, 5])
        ax2.grid(axis="y", alpha=0.3)
        
        # Compute mean and SD
        plaus_mean = np.mean(plaus_values)
        plaus_sd = np.std(plaus_values, ddof=1)
        ax2.text(0.98, 0.98, f"M = {plaus_mean:.2f}, SD = {plaus_sd:.2f}", 
                 transform=ax2.transAxes, ha="right", va="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.9), fontsize=9)
        
        print(f"Plausibility: M = {plaus_mean:.2f}, SD = {plaus_sd:.2f}")
    else:
        ax2.text(0.5, 0.5, "Plausibility data not available", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Distribution of Plausibility Ratings")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot to {out_path}")
    return out_path

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

if __name__ == "__main__":
    main()