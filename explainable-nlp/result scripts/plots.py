import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# TUM Color Palette
TUM_BLUE = "#0065BD"           # Primary blue
TUM_BLUE_DARK = "#005293"      # Secondary dark blue
TUM_BLUE_DARKER = "#003359"    # Secondary darker blue
TUM_ORANGE = "#E37222"         # Accent orange
TUM_GREEN = "#A2AD00"          # Accent green
TUM_LIGHT_BLUE = "#98C6EA"     # Accent light blue
TUM_MED_BLUE = "#64A0C8"       # Accent medium blue
TUM_BEIGE = "#DAD7CB"          # Accent beige
TUM_GRAY_80 = "#333333"        # 80% black
TUM_GRAY_50 = "#808080"        # 50% black
TUM_GRAY_20 = "#CCCCCC"        # 20% black
TUM_BLACK = "#000000"
TUM_WHITE = "#FFFFFF"

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
    sns.barplot(data=summary, x="faith_label", y="delta_conf", color=TUM_BLUE)
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
    sns.barplot(data=summary, x="faith_label", y="post_correct", color=TUM_MED_BLUE)
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
    sns.violinplot(data=df, x="faith_label", y="plaus", inner="box", cut=0, palette=[TUM_BLUE, TUM_ORANGE])
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
    
    plt.bar(x - width/2, before, width, label="Before (Initial)", color=TUM_BLUE, alpha=0.8)
    plt.bar(x + width/2, after, width, label="After (Post-explanation)", color=TUM_ORANGE, alpha=0.8)
    
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


def plot_per_question_accuracy_by_modelsize(long_df: pd.DataFrame, out_path: str = "per_question_accuracy_by_modelsize.png") -> str:
    """
    Plot accuracy per question showing:
    1. Initial accuracy (before seeing AI explanations)
    2. Small LLM accuracy (after seeing small LLM explanations)
    3. Large LLM accuracy (after seeing large LLM explanations)
    """
    df = long_df.copy()
    df["pre_correct"] = (df["pre"] == df["gt"]).astype(float)
    df["post_correct"] = (df["post"] == df["gt"]).astype(float)
    
    # Initial accuracy (same for all, averaged across all participants)
    initial_acc = df.groupby("Q")["pre_correct"].mean().reset_index()
    initial_acc.columns = ["Question", "Initial"]
    
    # Small LLM accuracy (model_size == 0)
    small_llm = df[df["model_size"] == 0].groupby("Q")["post_correct"].mean().reset_index()
    small_llm.columns = ["Question", "Small_LLM"]
    
    # Large LLM accuracy (model_size == 1)
    large_llm = df[df["model_size"] == 1].groupby("Q")["post_correct"].mean().reset_index()
    large_llm.columns = ["Question", "Large_LLM"]
    
    # Merge all together
    accuracy_by_q = initial_acc.merge(small_llm, on="Question", how="left").merge(large_llm, on="Question", how="left")
    
    # Plot
    plt.figure(figsize=(14, 6))
    x = np.arange(1, 17)
    width = 0.25
    
    initial = accuracy_by_q["Initial"].values
    small = accuracy_by_q["Small_LLM"].values
    large = accuracy_by_q["Large_LLM"].values
    
    plt.bar(x - width, initial, width, label="Initial (Before)", color=TUM_GRAY_50, alpha=0.8)
    plt.bar(x, small, width, label="Small LLM (Llama 3.1 8B)", color=TUM_LIGHT_BLUE, alpha=0.8)
    plt.bar(x + width, large, width, label="Large LLM (Llama 3.3 70B)", color=TUM_BLUE, alpha=0.8)
    
    # Add chance line at 0.5
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1.5, label="Chance (50%)")
    
    plt.xlabel("Question Number")
    plt.ylabel("Accuracy (Proportion Correct)")
    plt.title("Accuracy per Question: Initial vs. Small vs. Large LLM")
    plt.xticks(x)
    plt.ylim(0, 1)
    plt.legend(loc="best")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot to {out_path}")
    return out_path


def plot_per_question_accuracy_by_faithfulness(long_df: pd.DataFrame, out_path: str = "per_question_accuracy_by_faithfulness.png") -> str:
    """
    Plot accuracy per question showing:
    1. Initial accuracy (before seeing AI explanations)
    2. Faithful explanations accuracy (after seeing faithful explanations)
    3. Unfaithful explanations accuracy (after seeing unfaithful explanations)
    """
    df = long_df.copy()
    df["pre_correct"] = (df["pre"] == df["gt"]).astype(float)
    df["post_correct"] = (df["post"] == df["gt"]).astype(float)
    
    # Initial accuracy (same for all, averaged across all participants)
    initial_acc = df.groupby("Q")["pre_correct"].mean().reset_index()
    initial_acc.columns = ["Question", "Initial"]
    
    # Faithful explanations accuracy (faith == 1)
    faithful = df[df["faith"] == 1].groupby("Q")["post_correct"].mean().reset_index()
    faithful.columns = ["Question", "Faithful"]
    
    # Unfaithful explanations accuracy (faith == 0)
    unfaithful = df[df["faith"] == 0].groupby("Q")["post_correct"].mean().reset_index()
    unfaithful.columns = ["Question", "Unfaithful"]
    
    # Merge all together
    accuracy_by_q = initial_acc.merge(faithful, on="Question", how="left").merge(unfaithful, on="Question", how="left")
    
    # Plot
    plt.figure(figsize=(14, 6))
    x = np.arange(1, 17)
    width = 0.25
    
    initial = accuracy_by_q["Initial"].values
    faith = accuracy_by_q["Faithful"].values
    unfaith = accuracy_by_q["Unfaithful"].values
    
    plt.bar(x - width, initial, width, label="Initial (Before)", color=TUM_GRAY_50, alpha=0.8)
    plt.bar(x, faith, width, label="Faithful", color=TUM_GREEN, alpha=0.8)
    plt.bar(x + width, unfaith, width, label="Unfaithful", color=TUM_ORANGE, alpha=0.8)
    
    # Add chance line at 0.5
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1.5, label="Chance (50%)")
    
    plt.xlabel("Question Number")
    plt.ylabel("Accuracy (Proportion Correct)")
    plt.title("Accuracy per Question: Initial vs. Faithful vs. Unfaithful")
    plt.xticks(x)
    plt.ylim(0, 1)
    plt.legend(loc="best")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot to {out_path}")
    return out_path

def plot_plausibility_by_agreement(long_df: pd.DataFrame, out_path: str = "plausibility_by_agreement.png") -> str:
    """
    Plot plausibility ratings based on whether human and AI initially agreed or disagreed.
    Agreement: Human's initial answer (pre) matches AI's prediction (ai)
    Disagreement: Human's initial answer (pre) differs from AI's prediction (ai)
    """
    df = long_df.dropna(subset=["plaus", "pre", "ai"]).copy()
    
    # Create agreement variable: 1 if human and AI agree initially, 0 if they disagree
    df["agreement"] = (df["pre"] == df["ai"]).astype(int)
    
    # Calculate mean plausibility by agreement
    summary = df.groupby("agreement")["plaus"].agg(['mean', 'std', 'count']).reset_index()
    summary["agreement_label"] = summary["agreement"].map({1: "Agreement\n(Human = AI)", 0: "Disagreement\n(Human ≠ AI)"})
    
    # Calculate 95% CI
    summary['se'] = summary['std'] / np.sqrt(summary['count'])
    summary['ci'] = 1.96 * summary['se']
    
    # Plot
    plt.figure(figsize=(8, 6))
    x = np.arange(len(summary))
    
    bars = plt.bar(x, summary["mean"], color=[TUM_BLUE, TUM_ORANGE], alpha=0.8, edgecolor="black", linewidth=1.5)
    
    # Add error bars
    plt.errorbar(x, summary["mean"], yerr=summary["ci"], fmt='none', ecolor='black', capsize=5, capthick=2)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(summary.iterrows()):
        plt.text(i, row["mean"] + row["ci"] + 0.05, f"{row['mean']:.3f}", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add sample size labels
    for i, (idx, row) in enumerate(summary.iterrows()):
        plt.text(i, 0.3, f"n = {int(row['count'])}", 
                ha='center', va='bottom', fontsize=9, style='italic')
    
    plt.xticks(x, summary["agreement_label"])
    plt.ylabel("Mean Plausibility Rating (1-5)", fontsize=11)
    plt.xlabel("")
    plt.title("Plausibility: Agreement vs. Disagreement with AI", fontsize=12, fontweight='bold')
    plt.ylim(0, 5.5)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    
    # Print summary statistics
    print(f"\nPlausibility by Agreement:")
    print(f"  Agreement (Human = AI): M = {summary.loc[summary['agreement']==1, 'mean'].values[0]:.3f}, "
          f"SD = {summary.loc[summary['agreement']==1, 'std'].values[0]:.3f}, "
          f"n = {int(summary.loc[summary['agreement']==1, 'count'].values[0])}")
    print(f"  Disagreement (Human ≠ AI): M = {summary.loc[summary['agreement']==0, 'mean'].values[0]:.3f}, "
          f"SD = {summary.loc[summary['agreement']==0, 'std'].values[0]:.3f}, "
          f"n = {int(summary.loc[summary['agreement']==0, 'count'].values[0])}")
    
    print(f"Saved plot to {out_path}")
    return out_path


def plot_conf_change_by_agreement(long_df: pd.DataFrame, out_path: str = "conf_change_by_agreement.png") -> str:
    """
    Plot confidence change based on whether human and AI initially agreed or disagreed.
    Agreement: Human's initial answer (pre) matches AI's prediction (ai)
    Disagreement: Human's initial answer (pre) differs from AI's prediction (ai)
    """
    df = long_df.dropna(subset=["delta_conf", "pre", "ai"]).copy()
    
    # Create agreement variable: 1 if human and AI agree initially, 0 if they disagree
    df["agreement"] = (df["pre"] == df["ai"]).astype(int)
    
    # Calculate mean confidence change by agreement
    summary = df.groupby("agreement")["delta_conf"].agg(['mean', 'std', 'count']).reset_index()
    summary["agreement_label"] = summary["agreement"].map({1: "Agreement\n(Human = AI)", 0: "Disagreement\n(Human ≠ AI)"})
    
    # Calculate 95% CI
    summary['se'] = summary['std'] / np.sqrt(summary['count'])
    summary['ci'] = 1.96 * summary['se']
    
    # Plot
    plt.figure(figsize=(8, 6))
    x = np.arange(len(summary))
    
    bars = plt.bar(x, summary["mean"], color=[TUM_BLUE, TUM_ORANGE], alpha=0.8, edgecolor="black", linewidth=1.5)
    
    # Add error bars
    plt.errorbar(x, summary["mean"], yerr=summary["ci"], fmt='none', ecolor='black', capsize=5, capthick=2)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(summary.iterrows()):
        plt.text(i, row["mean"] + row["ci"] + 0.02, f"{row['mean']:.3f}", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add sample size labels
    for i, (idx, row) in enumerate(summary.iterrows()):
        plt.text(i, -0.05, f"n = {int(row['count'])}", 
                ha='center', va='top', fontsize=9, style='italic')
    
    plt.xticks(x, summary["agreement_label"])
    plt.ylabel("Mean Confidence Change (post - pre)", fontsize=11)
    plt.xlabel("")
    plt.title("Confidence Change: Agreement vs. Disagreement with AI", fontsize=12, fontweight='bold')
    plt.axhline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    
    # Print summary statistics
    print(f"\nConfidence Change by Agreement:")
    print(f"  Agreement (Human = AI): M = {summary.loc[summary['agreement']==1, 'mean'].values[0]:.3f}, "
          f"SD = {summary.loc[summary['agreement']==1, 'std'].values[0]:.3f}, "
          f"n = {int(summary.loc[summary['agreement']==1, 'count'].values[0])}")
    print(f"  Disagreement (Human ≠ AI): M = {summary.loc[summary['agreement']==0, 'mean'].values[0]:.3f}, "
          f"SD = {summary.loc[summary['agreement']==0, 'std'].values[0]:.3f}, "
          f"n = {int(summary.loc[summary['agreement']==0, 'count'].values[0])}")
    
    print(f"Saved plot to {out_path}")
    return out_path


def plot_aor_scatter_by_faith(long_df: pd.DataFrame, out_path: str = "aor_by_faith_scatter.png") -> str:
    """
    Plot RAIR (x-axis) vs RSR (y-axis) as two points: Faithful vs Unfaithful.
    Also annotate AOR for each group, defined as the mean of RAIR and RSR:
        AOR = (RAIR + RSR) / 2
    """
    # RAIR-eligible: AI correct & human initially wrong
    df_rair = long_df[(long_df["ai_correct"]==1) & (long_df["human_pre_correct"]==0)].copy()
    df_rair["rair"] = df_rair["changed_to_correct"].astype(float)
    rair_by_faith = df_rair.groupby("faith")["rair"].mean().rename("RAIR")

    # RSR-eligible: AI wrong & human initially correct
    df_rsr = long_df[(long_df["ai_correct"]==0) & (long_df["human_pre_correct"]==1)].copy()
    df_rsr["rsr"] = df_rsr["stayed_correct"].astype(float)
    rsr_by_faith = df_rsr.groupby("faith")["rsr"].mean().rename("RSR")

    # Combine
    summary = pd.concat([rair_by_faith, rsr_by_faith], axis=1).reset_index()  # columns: faith, RAIR, RSR
    summary["faith_label"] = summary["faith"].map({1: "Faithful", 0: "Unfaithful"})
    summary["AOR"] = (summary["RAIR"] + summary["RSR"]) / 2.0

    # Plot scatter
    plt.figure(figsize=(6, 6))
    # Use TUM blue palette for both groups to match requested styling
    colors = summary["faith"].map({1: TUM_BLUE, 0: TUM_LIGHT_BLUE}).values
    plt.scatter(summary["RAIR"], summary["RSR"], s=180, c=colors, edgecolor="black", linewidth=1.5)

    # Annotate each point with label and AOR
    for _, row in summary.iterrows():
        label = f"{row['faith_label']}\nAOR={row['AOR']:.3f}"
        # Place Faithful label below the dot to avoid overlap; Unfaithful above
        if int(row["faith"]) == 1:
            plt.annotate(label, (row["RAIR"], row["RSR"]), xytext=(8, -12), textcoords="offset points",
                         ha="left", va="top",
                         bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
        else:
            plt.annotate(label, (row["RAIR"], row["RSR"]), xytext=(8, 10), textcoords="offset points",
                         ha="left", va="bottom",
                         bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    # Formatting
    plt.xlabel("RAIR (AI correct & human initially wrong -> changed to correct)")
    plt.ylabel("RSR (AI wrong & human initially correct -> stayed correct)")
    plt.title("AOR Scatter by Faithfulness: RAIR (x) vs RSR (y)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)

    # Legend handles
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Faithful', markerfacecolor=TUM_BLUE, markeredgecolor='black', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Unfaithful', markerfacecolor=TUM_LIGHT_BLUE, markeredgecolor='black', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    # Print summary
    print("RAIR/RSR/AOR by Faithfulness:")
    for _, row in summary.iterrows():
        print(f"  {row['faith_label']}: RAIR={row['RAIR']:.3f}, RSR={row['RSR']:.3f}, AOR={row['AOR']:.3f}")

    print(f"Saved plot to {out_path}")
    return out_path


def plot_aor_scatter_by_modelsize(long_df: pd.DataFrame, out_path: str = "aor_by_modelsize_scatter.png") -> str:
    """
    Plot RAIR (x-axis) vs RSR (y-axis) as two points: Small vs Large LLMs.
    Annotate AOR = (RAIR + RSR)/2 for each.
    model_size: 0 = Small LLM, 1 = Large LLM
    """
    # RAIR-eligible subset
    df_rair = long_df[(long_df["ai_correct"]==1) & (long_df["human_pre_correct"]==0)].copy()
    df_rair["rair"] = df_rair["changed_to_correct"].astype(float)
    rair_by_size = df_rair.groupby("model_size")["rair"].mean().rename("RAIR")

    # RSR-eligible subset
    df_rsr = long_df[(long_df["ai_correct"]==0) & (long_df["human_pre_correct"]==1)].copy()
    df_rsr["rsr"] = df_rsr["stayed_correct"].astype(float)
    rsr_by_size = df_rsr.groupby("model_size")["rsr"].mean().rename("RSR")

    # Combine
    summary = pd.concat([rair_by_size, rsr_by_size], axis=1).reset_index()  # columns: model_size, RAIR, RSR
    summary["size_label"] = summary["model_size"].map({0: "Small LLM\n(Llama 3.1 8B)", 1: "Large LLM\n(Llama 3.3 70B)"})
    summary["AOR"] = (summary["RAIR"] + summary["RSR"]) / 2.0

    # Plot scatter with TUM blues
    plt.figure(figsize=(6, 6))
    colors = summary["model_size"].map({1: TUM_BLUE, 0: TUM_LIGHT_BLUE}).values
    plt.scatter(summary["RAIR"], summary["RSR"], s=180, c=colors, edgecolor="black", linewidth=1.5)

    # Annotate: Large above, Small below to avoid overlap
    for _, row in summary.iterrows():
        label = f"{row['size_label']}\nAOR={row['AOR']:.3f}"
        if int(row["model_size"]) == 0:
            plt.annotate(label, (row["RAIR"], row["RSR"]), xytext=(8, -12), textcoords="offset points",
                         ha="left", va="top",
                         bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
        else:
            plt.annotate(label, (row["RAIR"], row["RSR"]), xytext=(8, 10), textcoords="offset points",
                         ha="left", va="bottom",
                         bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    # Axes and layout
    plt.xlabel("RAIR (AI correct & human initially wrong -> changed to correct)")
    plt.ylabel("RSR (AI wrong & human initially correct -> stayed correct)")
    plt.title("AOR Scatter by Model Size: RAIR (x) vs RSR (y)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Large LLM', markerfacecolor=TUM_BLUE, markeredgecolor='black', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Small LLM', markerfacecolor=TUM_LIGHT_BLUE, markeredgecolor='black', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    # Print summary
    print("RAIR/RSR/AOR by Model Size:")
    for _, row in summary.iterrows():
        size = "Large" if int(row["model_size"]) == 1 else "Small"
        print(f"  {size}: RAIR={row['RAIR']:.3f}, RSR={row['RSR']:.3f}, AOR={row['AOR']:.3f}")

    print(f"Saved plot to {out_path}")
    return out_path


def plot_plausibility_vs_accuracy(long_df: pd.DataFrame, out_path: str = "plausibility_vs_accuracy.png") -> str:
    """
    Plot the relationship between plausibility ratings and final accuracy.
    Shows mean accuracy for each plausibility level (1-5) to see if more 
    plausible explanations lead to higher accuracy.
    """
    df = long_df.dropna(subset=["plaus", "post", "gt"]).copy()
    df["post_correct"] = (df["post"] == df["gt"]).astype(float)
    
    # Group by plausibility and compute mean accuracy
    summary = df.groupby("plaus")["post_correct"].agg(['mean', 'std', 'count']).reset_index()
    summary.columns = ["plausibility", "accuracy", "std", "count"]
    
    # Calculate 95% CI
    summary['se'] = summary['std'] / np.sqrt(summary['count'])
    summary['ci'] = 1.96 * summary['se']
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar plot with error bars
    bars = ax.bar(summary["plausibility"], summary["accuracy"], 
                   color=TUM_BLUE, alpha=0.7, edgecolor="black", linewidth=1.5)
    ax.errorbar(summary["plausibility"], summary["accuracy"], 
                yerr=summary["ci"], fmt='none', ecolor='black', capsize=5, capthick=2)
    
    # Add trend line
    z = np.polyfit(summary["plausibility"], summary["accuracy"], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(summary["plausibility"].min(), summary["plausibility"].max(), 100)
    ax.plot(x_trend, p(x_trend), color=TUM_ORANGE, linewidth=2.5, 
            linestyle="--", label=f"Trend: y={z[0]:.3f}x+{z[1]:.3f}")
    
    # Add value labels on bars
    for i, row in summary.iterrows():
        ax.text(row["plausibility"], row["accuracy"] + row["ci"] + 0.02, 
               f"{row['accuracy']:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add sample size labels
    for i, row in summary.iterrows():
        ax.text(row["plausibility"], 0.02, f"n={int(row['count'])}", 
               ha='center', va='bottom', fontsize=8, style='italic')
    
    ax.set_xlabel("Plausibility Rating (1 = Not at all plausible, 5 = Very plausible)", fontsize=11)
    ax.set_ylabel("Final Accuracy (Proportion Correct)", fontsize=11)
    ax.set_title("Relationship Between Plausibility and Final Accuracy", fontsize=12, fontweight='bold')
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.5, alpha=0.5, label="Chance (50%)")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    
    # Print summary
    print("\nPlausibility vs Accuracy:")
    for _, row in summary.iterrows():
        print(f"  Plausibility {int(row['plausibility'])}: Accuracy={row['accuracy']:.3f}, "
              f"SD={row['std']:.3f}, n={int(row['count'])}")
    
    # Correlation
    corr = df[["plaus", "post_correct"]].corr().iloc[0, 1]
    print(f"  Correlation (Pearson r): {corr:.3f}")
    
    print(f"Saved plot to {out_path}")
    return out_path


def plot_plausibility_vs_conf_change(long_df: pd.DataFrame, out_path: str = "plausibility_vs_conf_change.png") -> str:
    """
    Plot the relationship between plausibility ratings and confidence change.
    Shows mean confidence change for each plausibility level (1-5) to see if more 
    plausible explanations lead to larger confidence changes.
    """
    df = long_df.dropna(subset=["plaus", "delta_conf"]).copy()
    
    # Group by plausibility and compute mean confidence change
    summary = df.groupby("plaus")["delta_conf"].agg(['mean', 'std', 'count']).reset_index()
    summary.columns = ["plausibility", "conf_change", "std", "count"]
    
    # Calculate 95% CI
    summary['se'] = summary['std'] / np.sqrt(summary['count'])
    summary['ci'] = 1.96 * summary['se']
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar plot with error bars
    bars = ax.bar(summary["plausibility"], summary["conf_change"], 
                   color=TUM_MED_BLUE, alpha=0.7, edgecolor="black", linewidth=1.5)
    ax.errorbar(summary["plausibility"], summary["conf_change"], 
                yerr=summary["ci"], fmt='none', ecolor='black', capsize=5, capthick=2)
    
    # Add trend line
    z = np.polyfit(summary["plausibility"], summary["conf_change"], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(summary["plausibility"].min(), summary["plausibility"].max(), 100)
    ax.plot(x_trend, p(x_trend), color=TUM_ORANGE, linewidth=2.5, 
            linestyle="--", label=f"Trend: y={z[0]:.3f}x+{z[1]:.3f}")
    
    # Add value labels on bars
    for i, row in summary.iterrows():
        y_pos = row["conf_change"] + row["ci"] + 0.05 if row["conf_change"] >= 0 else row["conf_change"] - row["ci"] - 0.05
        va = 'bottom' if row["conf_change"] >= 0 else 'top'
        ax.text(row["plausibility"], y_pos, 
               f"{row['conf_change']:.3f}", ha='center', va=va, fontsize=9, fontweight='bold')
    
    # Add sample size labels at bottom
    y_min = min(summary["conf_change"] - summary["ci"])
    for i, row in summary.iterrows():
        ax.text(row["plausibility"], y_min - 0.1, f"n={int(row['count'])}", 
               ha='center', va='top', fontsize=8, style='italic')
    
    ax.set_xlabel("Plausibility Rating (1 = Not at all plausible, 5 = Very plausible)", fontsize=11)
    ax.set_ylabel("Mean Confidence Change (post - pre)", fontsize=11)
    ax.set_title("Relationship Between Plausibility and Confidence Change", fontsize=12, fontweight='bold')
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, label="No change")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    
    # Print summary
    print("\nPlausibility vs Confidence Change:")
    for _, row in summary.iterrows():
        print(f"  Plausibility {int(row['plausibility'])}: Conf Change={row['conf_change']:.3f}, "
              f"SD={row['std']:.3f}, n={int(row['count'])}")
    
    # Correlation
    corr = df[["plaus", "delta_conf"]].corr().iloc[0, 1]
    print(f"  Correlation (Pearson r): {corr:.3f}")
    
    print(f"Saved plot to {out_path}")
    return out_path


def plot_conf_vs_rair_scatter(long_df: pd.DataFrame, out_path: str = "conf_vs_rair_scatter.png") -> str:
    """
    Binned bar plot of confidence change (delta_conf) vs RAIR (changed_to_correct).
    Only includes RAIR-eligible trials (AI correct & human initially wrong).
    """
    # Filter to RAIR-eligible subset
    df = long_df[(long_df["ai_correct"]==1) & (long_df["human_pre_correct"]==0)].copy()
    
    if len(df) == 0:
        print("No RAIR-eligible data for plot")
        return out_path
    
    # Create bins for confidence change
    bins = [-10, -2, -1, 0, 1, 2, 10]  # Adjusted for typical confidence changes
    labels = ['≤-2', '-1', '0', '1', '2', '≥3']
    df['conf_bin'] = pd.cut(df['delta_conf'], bins=bins, labels=labels, include_lowest=True)
    
    # Calculate mean RAIR for each bin
    binned_rair = df.groupby('conf_bin', observed=True)['changed_to_correct'].agg(['mean', 'count']).reset_index()
    binned_rair.columns = ['conf_bin', 'mean_rair', 'count']
    
    plt.figure(figsize=(7, 5))
    
    bars = plt.bar(range(len(binned_rair)), binned_rair['mean_rair'], 
                   color=TUM_BLUE, alpha=0.8, edgecolor="black")
    
    # Add count labels on top of bars
    for i, (bar, count) in enumerate(zip(bars, binned_rair['count'])):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'n={int(count)}', ha='center', va='bottom', fontsize=8)
    
    plt.xticks(range(len(binned_rair)), binned_rair['conf_bin'])
    plt.xlabel("ΔConfidence (Post - Pre)")
    plt.ylabel("Mean RAIR (Proportion Changed to Correct)")
    plt.title("Confidence Change vs. RAIR (Binned)")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot to {out_path}")
    return out_path

def plot_conf_vs_rsr_scatter(long_df: pd.DataFrame, out_path: str = "conf_vs_rsr_scatter.png") -> str:
    """
    Binned bar plot of confidence change (delta_conf) vs RSR (stayed_correct).
    Only includes RSR-eligible trials (AI wrong & human initially correct).
    """
    # Filter to RSR-eligible subset
    df = long_df[(long_df["ai_correct"]==0) & (long_df["human_pre_correct"]==1)].copy()
    
    if len(df) == 0:
        print("No RSR-eligible data for plot")
        return out_path
    
    # Create bins for confidence change
    bins = [-10, -2, -1, 0, 1, 2, 10]  # Adjusted for typical confidence changes
    labels = ['≤-2', '-1', '0', '1', '2', '≥3']
    df['conf_bin'] = pd.cut(df['delta_conf'], bins=bins, labels=labels, include_lowest=True)
    
    # Calculate mean RSR for each bin
    binned_rsr = df.groupby('conf_bin', observed=True)['stayed_correct'].agg(['mean', 'count']).reset_index()
    binned_rsr.columns = ['conf_bin', 'mean_rsr', 'count']
    
    plt.figure(figsize=(7, 5))
    
    bars = plt.bar(range(len(binned_rsr)), binned_rsr['mean_rsr'], 
                   color=TUM_ORANGE, alpha=0.8, edgecolor="black")
    
    # Add count labels on top of bars
    for i, (bar, count) in enumerate(zip(bars, binned_rsr['count'])):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'n={int(count)}', ha='center', va='bottom', fontsize=8)
    
    plt.xticks(range(len(binned_rsr)), binned_rsr['conf_bin'])
    plt.xlabel("ΔConfidence (Post - Pre)")
    plt.ylabel("Mean RSR (Proportion Stayed Correct)")
    plt.title("Confidence Change vs. RSR (Binned)")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot to {out_path}")
    return out_path

def plot_rair_rsr_by_modelsize(long_df: pd.DataFrame, out_path: str = "rair_rsr_by_modelsize.png") -> str:
    """
    Bar plot showing mean RAIR and RSR by model size.
    RAIR is computed on RAIR-eligible subset, RSR on RSR-eligible subset.
    0 = Small LLM (Llama 3.1 8B), 1 = Large LLM (Llama 3.3 70B)
    """
    # RAIR-eligible: AI correct & human initially wrong
    df_rair = long_df[(long_df["ai_correct"]==1) & (long_df["human_pre_correct"]==0)].copy()
    df_rair["rair"] = df_rair["changed_to_correct"].astype(float)
    mean_rair = df_rair.groupby("model_size")["rair"].agg(['mean', 'count']).reset_index()
    mean_rair.columns = ["model_size", "mean_rair", "count_rair"]
    
    # RSR-eligible: AI wrong & human initially correct
    df_rsr = long_df[(long_df["ai_correct"]==0) & (long_df["human_pre_correct"]==1)].copy()
    df_rsr["rsr"] = df_rsr["stayed_correct"].astype(float)
    mean_rsr = df_rsr.groupby("model_size")["rsr"].agg(['mean', 'count']).reset_index()
    mean_rsr.columns = ["model_size", "mean_rsr", "count_rsr"]
    
    # Merge
    summary = pd.merge(mean_rair, mean_rsr, on="model_size", how="outer").fillna(0)
    summary['model_label'] = summary['model_size'].map({0: "Small LLM\n(Llama 3.1 8B)", 1: "Large LLM\n(Llama 3.3 70B)"})
    
    # Plot
    plt.figure(figsize=(8, 5))
    x = np.arange(len(summary))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, summary["mean_rair"], width, label="RAIR", color=TUM_BLUE, alpha=0.8, edgecolor="black")
    bars2 = plt.bar(x + width/2, summary["mean_rsr"], width, label="RSR", color=TUM_ORANGE, alpha=0.8, edgecolor="black")
    
    # Add value labels on bars
    for bar, val in zip(bars1, summary["mean_rair"]):
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    for bar, val in zip(bars2, summary["mean_rsr"]):
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.xticks(x, summary["model_label"])
    plt.ylabel("Mean Reliance (Proportion)")
    plt.xlabel("")
    plt.title("Mean RAIR and RSR by Model Size")
    plt.ylim(0, 1)
    plt.legend(loc="best")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot to {out_path}")
    
    # Print statistics
    for idx, row in summary.iterrows():
        model_name = row['model_label'].replace('\n', ' ')
        print(f"{model_name}:")
        print(f"  RAIR: M = {row['mean_rair']:.3f}, n = {int(row['count_rair'])}")
        print(f"  RSR: M = {row['mean_rsr']:.3f}, n = {int(row['count_rsr'])}")
    
    return out_path

def plot_conf_change_by_modelsize(long_df: pd.DataFrame, out_path: str = "conf_change_by_modelsize.png") -> str:
    """
    Bar plot of mean confidence change (delta_conf) by model size.
    0 = Small LLM (Llama 3.1 8B), 1 = Large LLM (Llama 3.3 70B)
    """
    df = long_df.dropna(subset=["delta_conf", "model_size"]).copy()
    
    # Group by model size and compute mean confidence change
    summary = df.groupby("model_size")["delta_conf"].agg(['mean', 'std', 'count']).reset_index()
    summary['model_label'] = summary['model_size'].map({0: "Small LLM\n(Llama 3.1 8B)", 1: "Large LLM\n(Llama 3.3 70B)"})
    
    plt.figure(figsize=(7, 5))
    
    bars = plt.bar(range(len(summary)), summary['mean'], 
                   color=[TUM_MED_BLUE, TUM_BLUE], alpha=0.8, edgecolor="black", width=0.6)
    
    # Add error bars (standard deviation)
    plt.errorbar(range(len(summary)), summary['mean'], yerr=summary['std'], 
                 fmt='none', ecolor='black', capsize=5, alpha=0.7)
    
    # Add value labels on top of bars
    for i, (bar, mean_val, count) in enumerate(zip(bars, summary['mean'], summary['count'])):
        height = bar.get_height()
        y_pos = height + summary['std'].iloc[i] + 0.05 if height > 0 else height - summary['std'].iloc[i] - 0.05
        va = 'bottom' if height > 0 else 'top'
        plt.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'M={mean_val:.2f}\nn={int(count)}', ha='center', va=va, fontsize=9)
    
    plt.xticks(range(len(summary)), summary['model_label'])
    plt.ylabel("Mean Confidence Change (Post - Pre)")
    plt.xlabel("")
    plt.title("Mean Change in Confidence by Model Size")
    plt.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot to {out_path}")
    
    # Print statistics
    for idx, row in summary.iterrows():
        model_name = row['model_label'].replace('\n', ' ')
        print(f"{model_name}: M = {row['mean']:.2f}, SD = {row['std']:.2f}, n = {int(row['count'])}")
    
    return out_path

def plot_plaus_vs_rair_rsr(long_df: pd.DataFrame, out_path: str = "plaus_vs_rair_rsr.png") -> str:
    """
    Binned bar plot showing effect of plausibility on RAIR and RSR.
    Groups plausibility into bins and computes mean RAIR/RSR for each bin.
    """
    # RAIR-eligible: AI correct & human initially wrong
    df_rair = long_df[(long_df["ai_correct"]==1) & (long_df["human_pre_correct"]==0)].copy()
    df_rair["rair"] = df_rair["changed_to_correct"].astype(float)
    
    # Group by plausibility rating
    rair_by_plaus = df_rair.groupby("plaus")["rair"].agg(['mean', 'count']).reset_index()
    rair_by_plaus.columns = ["plaus", "mean_rair", "count_rair"]
    
    # RSR-eligible: AI wrong & human initially correct
    df_rsr = long_df[(long_df["ai_correct"]==0) & (long_df["human_pre_correct"]==1)].copy()
    df_rsr["rsr"] = df_rsr["stayed_correct"].astype(float)
    
    # Group by plausibility rating
    rsr_by_plaus = df_rsr.groupby("plaus")["rsr"].agg(['mean', 'count']).reset_index()
    rsr_by_plaus.columns = ["plaus", "mean_rsr", "count_rsr"]
    
    # Merge on plausibility
    summary = pd.merge(rair_by_plaus, rsr_by_plaus, on="plaus", how="outer").fillna(0)
    summary = summary.sort_values("plaus")
    
    # Plot
    plt.figure(figsize=(10, 5))
    x = np.arange(len(summary))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, summary["mean_rair"], width, label="RAIR", color=TUM_BLUE, alpha=0.8, edgecolor="black")
    bars2 = plt.bar(x + width/2, summary["mean_rsr"], width, label="RSR", color=TUM_ORANGE, alpha=0.8, edgecolor="black")
    
    # Add value labels on bars
    for bar, val, count in zip(bars1, summary["mean_rair"], summary["count_rair"]):
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.2f}\n(n={int(count)})', ha='center', va='bottom', fontsize=7)
    
    for bar, val, count in zip(bars2, summary["mean_rsr"], summary["count_rsr"]):
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.2f}\n(n={int(count)})', ha='center', va='bottom', fontsize=7)
    
    plt.xticks(x, [f'{int(p)}' for p in summary["plaus"]])
    plt.xlabel("Plausibility Rating (1-5)")
    plt.ylabel("Mean Reliance (Proportion)")
    plt.title("Effect of Perceived Plausibility on RAIR and RSR")
    plt.ylim(0, 1.1)
    plt.legend(loc="best")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot to {out_path}")
    
    # Print statistics
    print("\nPlausibility vs RAIR/RSR:")
    for idx, row in summary.iterrows():
        print(f"Plausibility {int(row['plaus'])}:")
        print(f"  RAIR: M = {row['mean_rair']:.3f}, n = {int(row['count_rair'])}")
        print(f"  RSR: M = {row['mean_rsr']:.3f}, n = {int(row['count_rsr'])}")
    
    return out_path

def plot_accuracy_by_modelsize(long_df: pd.DataFrame, out_path: str = "accuracy_by_modelsize.png") -> str:
    """
    Bar plot of final accuracy (post-decision correctness) by model size.
    0 = Small LLM (Llama 3.1 8B), 1 = Large LLM (Llama 3.3 70B)
    """
    df = long_df.copy()
    df["post_correct"] = (df["post"] == df["gt"]).astype(float)
    
    # Group by model size and compute mean accuracy
    summary = df.groupby("model_size")["post_correct"].agg(['mean', 'std', 'count']).reset_index()
    summary['model_label'] = summary['model_size'].map({0: "Small LLM\n(Llama 3.1 8B)", 1: "Large LLM\n(Llama 3.3 70B)"})
    
    plt.figure(figsize=(7, 5))
    
    bars = plt.bar(range(len(summary)), summary['mean'], 
                   color=[TUM_MED_BLUE, TUM_BLUE], alpha=0.8, edgecolor="black", width=0.6)
    
    # Add error bars (standard deviation)
    plt.errorbar(range(len(summary)), summary['mean'], yerr=summary['std'], 
                 fmt='none', ecolor='black', capsize=5, alpha=0.7)
    
    # Add value labels on top of bars
    for i, (bar, mean_val, count) in enumerate(zip(bars, summary['mean'], summary['count'])):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + summary['std'].iloc[i] + 0.02,
                f'M={mean_val:.3f}\nn={int(count)}', ha='center', va='bottom', fontsize=9)
    
    plt.xticks(range(len(summary)), summary['model_label'])
    plt.ylabel("Final Accuracy (Proportion Correct)")
    plt.xlabel("")
    plt.title("Final Accuracy by Model Size")
    plt.ylim(0, 1)
    plt.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label="Chance (50%)")
    plt.legend(loc="upper right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot to {out_path}")
    
    # Print statistics
    for idx, row in summary.iterrows():
        model_name = row['model_label'].replace('\n', ' ')
        print(f"{model_name}: Accuracy M = {row['mean']:.3f}, SD = {row['std']:.3f}, n = {int(row['count'])}")
    
    return out_path

def plot_plausibility_by_modelsize(long_df: pd.DataFrame, out_path: str = "plausibility_by_modelsize.png") -> str:
    """
    Bar plot of mean plausibility ratings by model size.
    0 = Small LLM (Llama 3.1 8B), 1 = Large LLM (Llama 3.3 70B)
    """
    df = long_df.dropna(subset=["plaus", "model_size"]).copy()
    
    # Group by model size and compute mean plausibility
    summary = df.groupby("model_size")["plaus"].agg(['mean', 'std', 'count']).reset_index()
    summary['model_label'] = summary['model_size'].map({0: "Small LLM\n(Llama 3.1 8B)", 1: "Large LLM\n(Llama 3.3 70B)"})
    
    plt.figure(figsize=(7, 5))
    
    bars = plt.bar(range(len(summary)), summary['mean'], 
                   color=[TUM_MED_BLUE, TUM_BLUE], alpha=0.8, edgecolor="black", width=0.6)
    
    # Add error bars (standard deviation)
    plt.errorbar(range(len(summary)), summary['mean'], yerr=summary['std'], 
                 fmt='none', ecolor='black', capsize=5, alpha=0.7)
    
    # Add value labels on top of bars
    for i, (bar, mean_val, count) in enumerate(zip(bars, summary['mean'], summary['count'])):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + summary['std'].iloc[i] + 0.1,
                f'M={mean_val:.2f}\nn={int(count)}', ha='center', va='bottom', fontsize=9)
    
    plt.xticks(range(len(summary)), summary['model_label'])
    plt.ylabel("Mean Plausibility Rating (1–5)")
    plt.xlabel("")
    plt.title("Mean Plausibility Ratings by Model Size")
    plt.ylim(0, 5.5)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot to {out_path}")
    
    # Print statistics
    for idx, row in summary.iterrows():
        model_name = row['model_label'].replace('\n', ' ')
        print(f"{model_name}: M = {row['mean']:.2f}, SD = {row['std']:.2f}, n = {int(row['count'])}")
    
    return out_path

def plot_human_accuracy_before_after(long_df: pd.DataFrame, out_path: str = "human_accuracy_before_after.png") -> str:
    """
    Bar plot comparing initial human accuracy (before AI) vs final accuracy (after seeing AI).
    Shows overall improvement in human decision-making after exposure to AI explanations.
    """
    df = long_df.copy()
    
    # Compute pre and post accuracy
    df["pre_correct"] = (df["pre"] == df["gt"]).astype(float)
    df["post_correct"] = (df["post"] == df["gt"]).astype(float)
    
    # Calculate means and standard deviations
    pre_mean = df["pre_correct"].mean()
    post_mean = df["post_correct"].mean()
    
    pre_std = df["pre_correct"].std()
    post_std = df["post_correct"].std()
    
    n = len(df)
    
    # Create bar plot
    plt.figure(figsize=(7, 5))
    
    labels = ['Initial\n(Before AI)', 'Final\n(After AI)']
    means = [pre_mean, post_mean]
    stds = [pre_std, post_std]
    colors = [TUM_BLUE, TUM_ORANGE]
    
    bars = plt.bar(range(2), means, color=colors, alpha=0.8, edgecolor="black", width=0.6)
    
    # Add error bars (standard deviation)
    plt.errorbar(range(2), means, yerr=stds, 
                 fmt='none', ecolor='black', capsize=5, alpha=0.7)
    
    # Add value labels on top of bars
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.02,
                f'M={mean_val:.3f}\n(SD={std_val:.3f})', ha='center', va='bottom', fontsize=9)
    
    plt.xticks(range(2), labels)
    plt.ylabel("Accuracy (Proportion Correct)")
    plt.xlabel("")
    plt.title("Human Accuracy: Before vs After AI Explanations")
    plt.ylim(0, 1)
    
    # Add chance line
    plt.axhline(0.5, color=TUM_GRAY_50, linestyle='--', linewidth=1, alpha=0.7, label="Chance (50%)")
    
    # Add improvement annotation
    improvement = post_mean - pre_mean
    improvement_pct = (improvement / pre_mean) * 100
    plt.text(0.5, 0.05, f'Improvement: {improvement:+.3f} ({improvement_pct:+.1f}%)', 
             ha='center', fontsize=10, 
             bbox=dict(boxstyle="round", facecolor=TUM_BEIGE, alpha=0.8))
    
    plt.legend(loc="upper right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    
    print(f"Saved plot to {out_path}")
    print(f"Initial accuracy: M = {pre_mean:.3f}, SD = {pre_std:.3f}")
    print(f"Final accuracy: M = {post_mean:.3f}, SD = {post_std:.3f}")
    print(f"Improvement: {improvement:.3f} ({improvement_pct:+.1f}%)")
    print(f"Total observations: n = {n}")
    
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
        ax1.bar(x - width/2, conf1_freq, width, label="Before (Conf1)", color=TUM_BLUE, alpha=0.8, edgecolor="black")
        ax1.bar(x + width/2, conf2_freq, width, label="After (Conf2)", color=TUM_ORANGE, alpha=0.8, edgecolor="black")
        
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
        ax2.bar(plaus_counts.index, plaus_counts.values, color=TUM_MED_BLUE, alpha=0.8, edgecolor="black")
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

