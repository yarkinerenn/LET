import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# TUM color palette (single source of truth lives in config)
from ..config import (
    TUM_BLUE,
    TUM_BLUE_DARK,
    TUM_BLUE_DARKER,
    TUM_ORANGE,
    TUM_GREEN,
    TUM_LIGHT_BLUE,
    TUM_MED_BLUE,
    TUM_BEIGE,
    TUM_GRAY_80,
    TUM_GRAY_50,
    TUM_GRAY_20,
    TUM_BLACK,
    TUM_WHITE,
)

# Faithfulness / model-size codes as they are labelled in every plot
FAITH_LABELS = {1: "Faithful", 0: "Unfaithful"}
MODEL_SIZE_LABELS = {0: "Small LLM\n(Llama 3.1 8B)", 1: "Large LLM\n(Llama 3.3 70B)"}


def rair_eligible(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Trials the AI got right while the participant was initially wrong, with the
    binary RAIR outcome (did the participant switch to the correct answer?).
    """
    df = long_df[(long_df["ai_correct"] == 1) & (long_df["human_pre_correct"] == 0)].copy()
    df["rair"] = df["changed_to_correct"].astype(float)
    return df


def rsr_eligible(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Trials the AI got wrong while the participant was initially right, with the
    binary RSR outcome (did the participant stay correct?).
    """
    df = long_df[(long_df["ai_correct"] == 0) & (long_df["human_pre_correct"] == 1)].copy()
    df["rsr"] = df["stayed_correct"].astype(float)
    return df


def _write_figure(out_path: str) -> None:
    """Lay out, write and close the current figure."""
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _save(out_path: str) -> str:
    """Write the current figure and report where it went."""
    _write_figure(out_path)
    print(f"Saved plot to {out_path}")
    return out_path


def plot_mean_rair_rsr_by_faith(long_df: pd.DataFrame, out_path: str = "mean_rair_rsr_by_faith.png") -> str:
    """
    Compute mean RAIR (on RAIR-eligible subset) and mean RSR (on RSR-eligible subset)
    grouped by faithfulness (1=faithful, 0=unfaithful), and save a side-by-side bar plot.
    """
    df_rair = rair_eligible(long_df)
    mean_rair = df_rair.groupby("faith")["rair"].mean().rename("mean_rair")

    df_rsr = rsr_eligible(long_df)
    mean_rsr = df_rsr.groupby("faith")["rsr"].mean().rename("mean_rsr")

    summary = pd.concat([mean_rair, mean_rsr], axis=1)
    summary = summary.reset_index()  # columns: faith, mean_rair, mean_rsr
    summary["faith_label"] = summary["faith"].map(FAITH_LABELS)

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
    return _save(out_path)

def plot_mean_conf_change_by_faith(long_df: pd.DataFrame, out_path: str = "mean_conf_change_by_faith.png") -> str:
    """
    Plot average confidence change (post - pre) by faithfulness (1=faithful, 0=unfaithful).
    Uses the per-trial 'delta_conf' field from long_df.
    """
    df = long_df.dropna(subset=["delta_conf"]).copy()
    summary = df.groupby("faith")["delta_conf"].mean().reset_index()
    summary["faith_label"] = summary["faith"].map(FAITH_LABELS)

    plt.figure(figsize=(6,4))
    sns.barplot(data=summary, x="faith_label", y="delta_conf", color=TUM_BLUE)
    plt.ylabel("Mean Confidence Change (post - pre)")
    plt.xlabel("")
    plt.title("Average Confidence Change by Faithfulness")
    plt.axhline(0, color="gray", linewidth=1)
    return _save(out_path)

def plot_mean_final_accuracy_by_faith(long_df: pd.DataFrame, out_path: str = "mean_final_accuracy_by_faith.png") -> str:
    """
    Plot mean final task accuracy (post == gt) under faithful vs unfaithful explanations,
    averaged across model sizes (i.e., ignore model_size in grouping).
    """
    df = long_df.copy()
    df["post_correct"] = (df["post"] == df["gt"]).astype(float)
    summary = df.groupby("faith")["post_correct"].mean().reset_index()
    summary["faith_label"] = summary["faith"].map(FAITH_LABELS)

    plt.figure(figsize=(6,4))
    sns.barplot(data=summary, x="faith_label", y="post_correct", color=TUM_MED_BLUE)
    plt.ylabel("Mean Final Accuracy (post == GT)")
    plt.xlabel("")
    plt.title("Mean Final Task Accuracy by Faithfulness")
    plt.ylim(0,1)
    return _save(out_path)

def plot_plausibility_violin_by_faith(long_df: pd.DataFrame, out_path: str = "plausibility_violin_by_faith.png") -> str:
    """
    Violin plot of plausibility (1–5 Likert) for faithful vs unfaithful.
    """
    df = long_df.dropna(subset=["plaus"]).copy()
    df["faith_label"] = df["faith"].map(FAITH_LABELS)

    plt.figure(figsize=(7,4))
    sns.violinplot(data=df, x="faith_label", y="plaus", inner="box", cut=0, palette=[TUM_BLUE, TUM_ORANGE])
    plt.xlabel("")
    plt.ylabel("Plausibility (1–5)")
    plt.title("Plausibility by Faithfulness")
    plt.ylim(1,5)
    return _save(out_path)

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
    return _save(out_path)


def _plot_per_question_accuracy_split(long_df: pd.DataFrame, out_path: str, split_col: str,
                                      groups, title: str) -> str:
    """
    Accuracy per question: initial accuracy next to the post-explanation
    accuracy of each level of `split_col`.

    `groups` is a list of (value, label, color) for the two levels.
    """
    df = long_df.copy()
    df["pre_correct"] = (df["pre"] == df["gt"]).astype(float)
    df["post_correct"] = (df["post"] == df["gt"]).astype(float)

    questions = np.arange(1, 17)
    initial = df.groupby("Q")["pre_correct"].mean().reindex(questions)

    plt.figure(figsize=(14, 6))
    width = 0.25
    plt.bar(questions - width, initial.values, width, label="Initial (Before)",
            color=TUM_GRAY_50, alpha=0.8)

    for offset, (value, label, color) in zip([0, width], groups):
        accuracy = df[df[split_col] == value].groupby("Q")["post_correct"].mean().reindex(questions)
        plt.bar(questions + offset, accuracy.values, width, label=label, color=color, alpha=0.8)

    # Add chance line at 0.5
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1.5, label="Chance (50%)")

    plt.xlabel("Question Number")
    plt.ylabel("Accuracy (Proportion Correct)")
    plt.title(title)
    plt.xticks(questions)
    plt.ylim(0, 1)
    plt.legend(loc="best")
    plt.grid(axis="y", alpha=0.3)
    return _save(out_path)


def plot_per_question_accuracy_by_modelsize(long_df: pd.DataFrame, out_path: str = "per_question_accuracy_by_modelsize.png") -> str:
    """
    Plot accuracy per question showing:
    1. Initial accuracy (before seeing AI explanations)
    2. Small LLM accuracy (after seeing small LLM explanations)
    3. Large LLM accuracy (after seeing large LLM explanations)
    """
    return _plot_per_question_accuracy_split(
        long_df, out_path, "model_size",
        groups=[(0, "Small LLM (Llama 3.1 8B)", TUM_LIGHT_BLUE),
                (1, "Large LLM (Llama 3.3 70B)", TUM_BLUE)],
        title="Accuracy per Question: Initial vs. Small vs. Large LLM",
    )


def plot_per_question_accuracy_by_faithfulness(long_df: pd.DataFrame, out_path: str = "per_question_accuracy_by_faithfulness.png") -> str:
    """
    Plot accuracy per question showing:
    1. Initial accuracy (before seeing AI explanations)
    2. Faithful explanations accuracy (after seeing faithful explanations)
    3. Unfaithful explanations accuracy (after seeing unfaithful explanations)
    """
    return _plot_per_question_accuracy_split(
        long_df, out_path, "faith",
        groups=[(1, "Faithful", TUM_GREEN), (0, "Unfaithful", TUM_ORANGE)],
        title="Accuracy per Question: Initial vs. Faithful vs. Unfaithful",
    )


def plot_rair_rsr_per_question(long_df: pd.DataFrame, out_path: str = "rair_rsr_per_question.png") -> str:
    """
    Plot RAIR and RSR per question across all participants.
    RAIR: Proportion who changed to correct (among RAIR-eligible trials)
    RSR: Proportion who stayed correct (among RSR-eligible trials)
    Shows how reliance varies across different questions.
    """
    questions = sorted(long_df['Q'].unique())
    
    rair_values = []
    rsr_values = []
    rair_counts = []
    rsr_counts = []
    
    for q in questions:
        q_data = long_df[long_df['Q'] == q]
        
        # RAIR: AI correct & human initially wrong
        rair_eligible = q_data[(q_data['ai_correct'] == 1) & (q_data['human_pre_correct'] == 0)]
        if len(rair_eligible) > 0:
            rair = rair_eligible['changed_to_correct'].mean()
            rair_values.append(rair)
            rair_counts.append(len(rair_eligible))
        else:
            rair_values.append(np.nan)
            rair_counts.append(0)
        
        # RSR: AI wrong & human initially correct
        rsr_eligible = q_data[(q_data['ai_correct'] == 0) & (q_data['human_pre_correct'] == 1)]
        if len(rsr_eligible) > 0:
            rsr = rsr_eligible['stayed_correct'].mean()
            rsr_values.append(rsr)
            rsr_counts.append(len(rsr_eligible))
        else:
            rsr_values.append(np.nan)
            rsr_counts.append(0)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    x = np.array(questions)
    
    # Top panel: RAIR
    bars1 = ax1.bar(x, rair_values, color=TUM_BLUE, alpha=0.8, edgecolor="black")
    ax1.set_ylabel("RAIR (Proportion Changed to Correct)")
    ax1.set_title("RAIR per Question: Reliance on AI when AI is Correct")
    ax1.set_ylim(0, 1)
    ax1.grid(axis="y", alpha=0.3)
    ax1.axhline(0.5, color=TUM_GRAY_50, linestyle='--', linewidth=1, alpha=0.5)
    
    # Add count labels on RAIR bars
    for i, (bar, count) in enumerate(zip(bars1, rair_counts)):
        if not np.isnan(rair_values[i]) and count > 0:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'n={count}', ha='center', va='bottom', fontsize=7)
    
    # Bottom panel: RSR
    bars2 = ax2.bar(x, rsr_values, color=TUM_ORANGE, alpha=0.8, edgecolor="black")
    ax2.set_ylabel("RSR (Proportion Stayed Correct)")
    ax2.set_xlabel("Question Number")
    ax2.set_title("RSR per Question: Resistance to AI when AI is Wrong")
    ax2.set_ylim(0, 1)
    ax2.set_xticks(x)
    ax2.grid(axis="y", alpha=0.3)
    ax2.axhline(0.5, color=TUM_GRAY_50, linestyle='--', linewidth=1, alpha=0.5)
    
    # Add count labels on RSR bars
    for i, (bar, count) in enumerate(zip(bars2, rsr_counts)):
        if not np.isnan(rsr_values[i]) and count > 0:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'n={count}', ha='center', va='bottom', fontsize=7)
    
    _write_figure(out_path)
    
    print(f"Saved plot to {out_path}")
    
    # Print summary statistics
    rair_mean = np.nanmean(rair_values)
    rsr_mean = np.nanmean(rsr_values)
    print(f"\nRAIR per question: M = {rair_mean:.3f} (across {sum(1 for x in rair_values if not np.isnan(x))} questions with eligible trials)")
    print(f"RSR per question: M = {rsr_mean:.3f} (across {sum(1 for x in rsr_values if not np.isnan(x))} questions with eligible trials)")
    
    return out_path


AGREEMENT_LABELS = {1: "Agreement\n(Human = AI)", 0: "Disagreement\n(Human ≠ AI)"}


def _plot_by_agreement(long_df: pd.DataFrame, out_path: str, value_col: str, ylabel: str,
                       title: str, summary_title: str, value_label_offset: float,
                       n_label_y: float, n_label_va: str, ylim=None, zero_line: bool = False) -> str:
    """
    Bar plot of `value_col` for trials where the participant's initial answer
    agreed with the AI versus trials where it did not, with 95% CI error bars,
    value labels and cell sizes.
    """
    df = long_df.dropna(subset=[value_col, "pre", "ai"]).copy()
    df["agreement"] = (df["pre"] == df["ai"]).astype(int)

    summary = df.groupby("agreement")[value_col].agg(["mean", "std", "count"]).reset_index()
    summary["agreement_label"] = summary["agreement"].map(AGREEMENT_LABELS)
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    summary["ci"] = 1.96 * summary["se"]

    plt.figure(figsize=(8, 6))
    x = np.arange(len(summary))
    plt.bar(x, summary["mean"], color=[TUM_BLUE, TUM_ORANGE], alpha=0.8,
            edgecolor="black", linewidth=1.5)
    plt.errorbar(x, summary["mean"], yerr=summary["ci"], fmt="none",
                 ecolor="black", capsize=5, capthick=2)

    for i, (_, row) in enumerate(summary.iterrows()):
        plt.text(i, row["mean"] + row["ci"] + value_label_offset, f"{row['mean']:.3f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
        plt.text(i, n_label_y, f"n = {int(row['count'])}",
                 ha="center", va=n_label_va, fontsize=9, style="italic")

    plt.xticks(x, summary["agreement_label"])
    plt.ylabel(ylabel, fontsize=11)
    plt.xlabel("")
    plt.title(title, fontsize=12, fontweight="bold")
    if ylim is not None:
        plt.ylim(*ylim)
    if zero_line:
        plt.axhline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    plt.grid(axis="y", alpha=0.3)
    _write_figure(out_path)

    print(f"\n{summary_title}:")
    for value, label in [(1, "Agreement (Human = AI)"), (0, "Disagreement (Human ≠ AI)")]:
        row = summary[summary["agreement"] == value].iloc[0]
        print(f"  {label}: M = {row['mean']:.3f}, SD = {row['std']:.3f}, n = {int(row['count'])}")

    print(f"Saved plot to {out_path}")
    return out_path


def plot_plausibility_by_agreement(long_df: pd.DataFrame, out_path: str = "plausibility_by_agreement.png") -> str:
    """
    Plot plausibility ratings based on whether human and AI initially agreed.
    Agreement: the participant's initial answer (pre) matches the AI's prediction.
    """
    return _plot_by_agreement(
        long_df, out_path, value_col="plaus",
        ylabel="Mean Plausibility Rating (1-5)",
        title="Plausibility: Agreement vs. Disagreement with AI",
        summary_title="Plausibility by Agreement",
        value_label_offset=0.05, n_label_y=0.3, n_label_va="bottom", ylim=(0, 5.5),
    )


def plot_conf_change_by_agreement(long_df: pd.DataFrame, out_path: str = "conf_change_by_agreement.png") -> str:
    """
    Plot confidence change based on whether human and AI initially agreed.
    Agreement: the participant's initial answer (pre) matches the AI's prediction.
    """
    return _plot_by_agreement(
        long_df, out_path, value_col="delta_conf",
        ylabel="Mean Confidence Change (post - pre)",
        title="Confidence Change: Agreement vs. Disagreement with AI",
        summary_title="Confidence Change by Agreement",
        value_label_offset=0.02, n_label_y=-0.05, n_label_va="top", zero_line=True,
    )


def _plot_aor_scatter(long_df: pd.DataFrame, out_path: str, group_col: str, labels: dict,
                      colors: dict, label_below: int, legend_order, title: str,
                      summary_title: str, summary_labels: dict) -> str:
    """
    RAIR (x) against RSR (y) as one point per level of `group_col`, annotated
    with that group's AOR = (RAIR + RSR) / 2.

    `label_below` is the level whose annotation goes under its point so the two
    labels do not overlap.
    """
    rair = rair_eligible(long_df).groupby(group_col)["rair"].mean().rename("RAIR")
    rsr = rsr_eligible(long_df).groupby(group_col)["rsr"].mean().rename("RSR")

    summary = pd.concat([rair, rsr], axis=1).reset_index()
    summary["label"] = summary[group_col].map(labels)
    summary["AOR"] = (summary["RAIR"] + summary["RSR"]) / 2.0

    plt.figure(figsize=(6, 6))
    plt.scatter(summary["RAIR"], summary["RSR"], s=180,
                c=summary[group_col].map(colors).values, edgecolor="black", linewidth=1.5)

    for _, row in summary.iterrows():
        below = int(row[group_col]) == label_below
        plt.annotate(f"{row['label']}\nAOR={row['AOR']:.3f}", (row["RAIR"], row["RSR"]),
                     xytext=(8, -12) if below else (8, 10), textcoords="offset points",
                     ha="left", va="top" if below else "bottom",
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    plt.xlabel("RAIR (AI correct & human initially wrong -> changed to correct)")
    plt.ylabel("RSR (AI wrong & human initially correct -> stayed correct)")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)

    plt.legend(handles=[Line2D([0], [0], marker="o", color="w", label=label,
                               markerfacecolor=colors[value], markeredgecolor="black",
                               markersize=10)
                        for value, label in legend_order],
               loc="lower right")

    _write_figure(out_path)

    print(f"{summary_title}:")
    for _, row in summary.iterrows():
        print(f"  {summary_labels[int(row[group_col])]}: RAIR={row['RAIR']:.3f}, "
              f"RSR={row['RSR']:.3f}, AOR={row['AOR']:.3f}")

    print(f"Saved plot to {out_path}")
    return out_path


def plot_aor_scatter_by_faith(long_df: pd.DataFrame, out_path: str = "aor_by_faith_scatter.png") -> str:
    """
    Plot RAIR (x-axis) vs RSR (y-axis) as two points: Faithful vs Unfaithful.
    Also annotate AOR for each group, defined as the mean of RAIR and RSR:
        AOR = (RAIR + RSR) / 2
    """
    return _plot_aor_scatter(
        long_df, out_path, group_col="faith", labels=FAITH_LABELS,
        colors={1: TUM_BLUE, 0: TUM_LIGHT_BLUE}, label_below=1,
        legend_order=[(1, "Faithful"), (0, "Unfaithful")],
        title="AOR Scatter by Faithfulness: RAIR (x) vs RSR (y)",
        summary_title="RAIR/RSR/AOR by Faithfulness", summary_labels=FAITH_LABELS,
    )


def plot_aor_scatter_by_modelsize(long_df: pd.DataFrame, out_path: str = "aor_by_modelsize_scatter.png") -> str:
    """
    Plot RAIR (x-axis) vs RSR (y-axis) as two points: Small vs Large LLMs.
    Annotate AOR = (RAIR + RSR)/2 for each.
    model_size: 0 = Small LLM, 1 = Large LLM
    """
    return _plot_aor_scatter(
        long_df, out_path, group_col="model_size",
        labels=MODEL_SIZE_LABELS,
        colors={1: TUM_BLUE, 0: TUM_LIGHT_BLUE}, label_below=0,
        legend_order=[(1, "Large LLM"), (0, "Small LLM")],
        title="AOR Scatter by Model Size: RAIR (x) vs RSR (y)",
        summary_title="RAIR/RSR/AOR by Model Size",
        summary_labels={1: "Large", 0: "Small"},
    )


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
    
    _write_figure(out_path)
    
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
    
    _write_figure(out_path)
    
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


# Confidence-change bins used by the binned reliance plots
CONF_CHANGE_BINS = [-10, -2, -1, 0, 1, 2, 10]
CONF_CHANGE_BIN_LABELS = ["≤-2", "-1", "0", "1", "2", "≥3"]


def _plot_reliance_by_conf_bin(eligible_df: pd.DataFrame, out_path: str, outcome: str,
                               empty_message: str, color: str, ylabel: str, title: str) -> str:
    """
    Mean reliance outcome per bin of confidence change, with the cell size on
    top of each bar.
    """
    if len(eligible_df) == 0:
        print(empty_message)
        return out_path

    df = eligible_df.copy()
    df["conf_bin"] = pd.cut(df["delta_conf"], bins=CONF_CHANGE_BINS,
                            labels=CONF_CHANGE_BIN_LABELS, include_lowest=True)
    binned = df.groupby("conf_bin", observed=True)[outcome].agg(["mean", "count"]).reset_index()

    plt.figure(figsize=(7, 5))
    bars = plt.bar(range(len(binned)), binned["mean"], color=color, alpha=0.8, edgecolor="black")

    for bar, count in zip(bars, binned["count"]):
        plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                 f"n={int(count)}", ha="center", va="bottom", fontsize=8)

    plt.xticks(range(len(binned)), binned["conf_bin"])
    plt.xlabel("ΔConfidence (Post - Pre)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    return _save(out_path)


def plot_conf_vs_rair_scatter(long_df: pd.DataFrame, out_path: str = "conf_vs_rair_scatter.png") -> str:
    """
    Binned bar plot of confidence change (delta_conf) vs RAIR (changed_to_correct).
    Only includes RAIR-eligible trials (AI correct & human initially wrong).
    """
    return _plot_reliance_by_conf_bin(
        rair_eligible(long_df), out_path, outcome="changed_to_correct",
        empty_message="No RAIR-eligible data for plot", color=TUM_BLUE,
        ylabel="Mean RAIR (Proportion Changed to Correct)",
        title="Confidence Change vs. RAIR (Binned)",
    )


def plot_conf_vs_rsr_scatter(long_df: pd.DataFrame, out_path: str = "conf_vs_rsr_scatter.png") -> str:
    """
    Binned bar plot of confidence change (delta_conf) vs RSR (stayed_correct).
    Only includes RSR-eligible trials (AI wrong & human initially correct).
    """
    return _plot_reliance_by_conf_bin(
        rsr_eligible(long_df), out_path, outcome="stayed_correct",
        empty_message="No RSR-eligible data for plot", color=TUM_ORANGE,
        ylabel="Mean RSR (Proportion Stayed Correct)",
        title="Confidence Change vs. RSR (Binned)",
    )


def plot_rair_rsr_by_modelsize(long_df: pd.DataFrame, out_path: str = "rair_rsr_by_modelsize.png") -> str:
    """
    Bar plot showing mean RAIR and RSR by model size.
    RAIR is computed on RAIR-eligible subset, RSR on RSR-eligible subset.
    0 = Small LLM (Llama 3.1 8B), 1 = Large LLM (Llama 3.3 70B)
    """
    df_rair = rair_eligible(long_df)
    mean_rair = df_rair.groupby("model_size")["rair"].agg(['mean', 'count']).reset_index()
    mean_rair.columns = ["model_size", "mean_rair", "count_rair"]
    
    df_rsr = rsr_eligible(long_df)
    mean_rsr = df_rsr.groupby("model_size")["rsr"].agg(['mean', 'count']).reset_index()
    mean_rsr.columns = ["model_size", "mean_rsr", "count_rsr"]
    
    # Merge
    summary = pd.merge(mean_rair, mean_rsr, on="model_size", how="outer").fillna(0)
    summary['model_label'] = summary['model_size'].map(MODEL_SIZE_LABELS)
    
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
    _write_figure(out_path)
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
    summary['model_label'] = summary['model_size'].map(MODEL_SIZE_LABELS)
    
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
    _write_figure(out_path)
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
    df_rair = rair_eligible(long_df)
    
    # Group by plausibility rating
    rair_by_plaus = df_rair.groupby("plaus")["rair"].agg(['mean', 'count']).reset_index()
    rair_by_plaus.columns = ["plaus", "mean_rair", "count_rair"]
    
    df_rsr = rsr_eligible(long_df)
    
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
    _write_figure(out_path)
    print(f"Saved plot to {out_path}")
    
    # Print statistics
    print("\nPlausibility vs RAIR/RSR:")
    for idx, row in summary.iterrows():
        print(f"Plausibility {int(row['plaus'])}:")
        print(f"  RAIR: M = {row['mean_rair']:.3f}, n = {int(row['count_rair'])}")
        print(f"  RSR: M = {row['mean_rsr']:.3f}, n = {int(row['count_rsr'])}")
    
    return out_path

def _plot_mean_by_modelsize(df: pd.DataFrame, out_path: str, value_col: str, ylabel: str,
                            title: str, ylim, decimals: int, label_offset: float,
                            stat_label: str, chance_line: bool = False) -> str:
    """
    Bar plot of the mean of `value_col` per model size, with SD error bars and
    mean/cell-size labels, followed by the same numbers on the console.
    """
    summary = df.groupby("model_size")[value_col].agg(["mean", "std", "count"]).reset_index()
    summary["model_label"] = summary["model_size"].map(MODEL_SIZE_LABELS)

    plt.figure(figsize=(7, 5))
    bars = plt.bar(range(len(summary)), summary["mean"],
                   color=[TUM_MED_BLUE, TUM_BLUE], alpha=0.8, edgecolor="black", width=0.6)
    plt.errorbar(range(len(summary)), summary["mean"], yerr=summary["std"],
                 fmt="none", ecolor="black", capsize=5, alpha=0.7)

    for bar, (_, row) in zip(bars, summary.iterrows()):
        plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + row["std"] + label_offset,
                 f"M={row['mean']:.{decimals}f}\nn={int(row['count'])}",
                 ha="center", va="bottom", fontsize=9)

    plt.xticks(range(len(summary)), summary["model_label"])
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.title(title)
    plt.ylim(*ylim)
    if chance_line:
        plt.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Chance (50%)")
        plt.legend(loc="upper right")
    plt.grid(axis="y", alpha=0.3)
    _write_figure(out_path)
    print(f"Saved plot to {out_path}")

    for _, row in summary.iterrows():
        model_name = row["model_label"].replace("\n", " ")
        print(f"{model_name}: {stat_label}M = {row['mean']:.{decimals}f}, "
              f"SD = {row['std']:.{decimals}f}, n = {int(row['count'])}")

    return out_path


def plot_accuracy_by_modelsize(long_df: pd.DataFrame, out_path: str = "accuracy_by_modelsize.png") -> str:
    """
    Bar plot of final accuracy (post-decision correctness) by model size.
    0 = Small LLM (Llama 3.1 8B), 1 = Large LLM (Llama 3.3 70B)
    """
    df = long_df.copy()
    df["post_correct"] = (df["post"] == df["gt"]).astype(float)
    return _plot_mean_by_modelsize(
        df, out_path, value_col="post_correct",
        ylabel="Final Accuracy (Proportion Correct)", title="Final Accuracy by Model Size",
        ylim=(0, 1), decimals=3, label_offset=0.02, stat_label="Accuracy ", chance_line=True,
    )


def plot_plausibility_by_modelsize(long_df: pd.DataFrame, out_path: str = "plausibility_by_modelsize.png") -> str:
    """
    Bar plot of mean plausibility ratings by model size.
    0 = Small LLM (Llama 3.1 8B), 1 = Large LLM (Llama 3.3 70B)
    """
    return _plot_mean_by_modelsize(
        long_df.dropna(subset=["plaus", "model_size"]), out_path, value_col="plaus",
        ylabel="Mean Plausibility Rating (1–5)", title="Mean Plausibility Ratings by Model Size",
        ylim=(0, 5.5), decimals=2, label_offset=0.1, stat_label="",
    )


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
    _write_figure(out_path)
    
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
    
    return _save(out_path)

