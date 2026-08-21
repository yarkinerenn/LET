"""
Shared building blocks for the demographic analyses.

The five demographic scripts (age, gender, CS/AI expertise, education, NLP
experience) all follow the same recipe:

1. load the processed wide-format data and add the demographic to the
   trial-level (long) data,
2. aggregate to participant level,
3. describe the relationship (Spearman correlations for ordinal demographics,
   descriptive statistics for categorical ones),
4. run the same five regressions with cluster-robust standard errors,
5. draw the same four kinds of plot.

Everything that does not depend on the specific demographic lives here; the
individual scripts only supply columns, labels and colors.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

from .. import config
from ..analysis.hypothesis_tests import make_long, ols_clustered, logit_clustered
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
)

# Participant-level metrics analysed by every demographic script
METRICS = ["RAIR_user", "RSR_user", "Mean_Plausibility", "Mean_Conf_Delta", "Final_Accuracy_User"]
METRIC_LABELS = ["RAIR", "RSR", "Mean Plausibility", "Mean Confidence Change", "Final Accuracy"]
METRIC_COLORS = [TUM_BLUE, TUM_ORANGE, TUM_GREEN, TUM_MED_BLUE, TUM_BEIGE]

DATA_FILE = config.PROCESSED_DATA_FILE

BOX_STYLE = dict(
    boxprops=dict(facecolor=TUM_LIGHT_BLUE, alpha=0.7, edgecolor=TUM_BLUE_DARK),
    whiskerprops=dict(color=TUM_BLUE_DARK, linewidth=1.5),
    capprops=dict(color=TUM_BLUE_DARK, linewidth=1.5),
    medianprops=dict(color=TUM_ORANGE, linewidth=2.5),
    flierprops=dict(marker="o", markerfacecolor=TUM_GRAY_50, markersize=6, alpha=0.5),
)


# ============================================================================
# SMALL HELPERS
# ============================================================================

def sig_stars(p_value, ns=""):
    """Significance marker for a p-value ('' or 'ns' when not significant)."""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ns


def effect_size_label(rho):
    """Cohen's guidelines for the size of a correlation."""
    abs_rho = abs(rho)
    if abs_rho < 0.1:
        return "Negligible"
    if abs_rho < 0.3:
        return "Weak"
    if abs_rho < 0.5:
        return "Moderate"
    return "Strong"


def _save(fig_path):
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")


def _annotate_spearman(ax, x, y, n_label="Total n"):
    """Bottom-right box with the Spearman correlation of the plotted data."""
    rho, p_value = stats.spearmanr(x, y)
    ax.text(
        0.98, 0.02,
        f"Spearman ρ = {rho:.3f} {sig_stars(p_value, ns='ns')}\np = {p_value:.4f}\n{n_label} = {len(x)}",
        transform=ax.transAxes, fontsize=11, verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor=TUM_GRAY_50),
    )


# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_wide_data(title, column, column_hints):
    """
    Print the analysis header, create the output folder and load the processed
    data. Returns None if the file or the demographic column is missing.
    """
    print("=" * 80)
    print(title)
    print("=" * 80)
    print(f"\nLoading data from: {DATA_FILE}")
    print("Note: Always uses unfiltered data (all participants)\n")

    try:
        df_wide = pd.read_excel(str(DATA_FILE))
        print(f"Loaded {len(df_wide)} participants")
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found. Please run the data processing step first.")
        return None

    if column not in df_wide.columns:
        print(f"\nError: Column '{column}' not found in data.")
        print(f"Available columns containing {' or '.join(repr(h) for h in column_hints)}:")
        for col in df_wide.columns:
            if any(hint in col.lower() for hint in column_hints):
                print(f"  - {col}")
        return None

    return df_wide


def prepare_output_folder(output_folder):
    """Create the plot output folder if it does not exist yet."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}/")
    else:
        print(f"Using output folder: {output_folder}/")


def build_long_data(df_wide):
    """Expand the wide data into one row per trial."""
    print("\nConverting to long format...")
    long_df = make_long(df_wide, n_trials=config.N_TRIALS)
    print(f"Created {len(long_df)} trial observations")
    return long_df


def map_to_long(long_df, df_wide, source_col, target_col, mapping=None):
    """Copy a participant-level column onto every trial of that participant."""
    values = df_wide[source_col]
    if mapping is not None:
        values = values.map(mapping)
    long_df[target_col] = long_df["participant"].map(values.to_dict())
    return long_df


def compute_participant_level_metrics(long_df, df_wide, group_cols):
    """
    Aggregate trial-level metrics to participant level and merge them with the
    participant-level metrics already present in the wide data.

    `group_cols` are the demographic columns carried over unchanged.
    """
    aggregations = {"plaus": "mean", "delta_conf": "mean"}
    aggregations.update({col: "first" for col in group_cols})

    participant_agg = (
        long_df.groupby("participant")
        .agg(aggregations)
        .reset_index()
        .rename(columns={"plaus": "Mean_Plausibility", "delta_conf": "Mean_Conf_Delta"})
    )

    wide_metrics = df_wide[["RAIR_user", "RSR_user", "Final_Accuracy_User"]].copy()
    wide_metrics["participant"] = wide_metrics.index

    return participant_agg.merge(wide_metrics, on="participant", how="left")


# ============================================================================
# CORRELATIONS
# ============================================================================

def spearman_correlations(participant_df, predictor_col, metrics=METRICS):
    """Spearman rank correlations between the demographic and each metric."""
    results = []

    for metric in metrics:
        valid_data = participant_df[[predictor_col, metric]].dropna()

        if len(valid_data) < 3:
            results.append({
                "Metric": metric,
                "Spearman_rho": np.nan,
                "p_value": np.nan,
                "n": len(valid_data),
                "Significance": "",
                "Effect_Size": "Insufficient data",
            })
            continue

        rho, p_value = stats.spearmanr(valid_data[predictor_col], valid_data[metric])
        results.append({
            "Metric": metric,
            "Spearman_rho": rho,
            "p_value": p_value,
            "n": len(valid_data),
            "Significance": sig_stars(p_value),
            "Effect_Size": effect_size_label(rho),
        })

    return pd.DataFrame(results)


def print_correlation_summary(results_df, title):
    """Print the correlation table to the console."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print("\nSpearman Rank Correlations (Non-parametric)")
    print("Significance: * p<.05, ** p<.01, *** p<.001")
    print("Effect Size: |ρ| < 0.1 (Negligible), 0.1-0.3 (Weak), 0.3-0.5 (Moderate), > 0.5 (Strong)")
    print("\n" + "-" * 80)

    for _, row in results_df.iterrows():
        print(f"\n{row['Metric']}:")
        print(f"  Spearman ρ = {row['Spearman_rho']:.4f} {row['Significance']}")
        print(f"  p-value = {row['p_value']:.4f}")
        print(f"  n = {row['n']}")
        print(f"  Effect size: {row['Effect_Size']}")

    print("\n" + "=" * 80)


def save_correlation_summary(results_df, output_path):
    """Write the correlation table to CSV."""
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved correlation summary to: {output_path}")


def print_descriptive_statistics(participant_df, group_col, title, metrics=METRICS):
    """Print mean/std/count of every metric per group."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for metric in metrics:
        print(f"\n{metric}:")
        print(participant_df.groupby(group_col)[metric].agg(["mean", "std", "count"]).to_string())


# ============================================================================
# PLOTS
# ============================================================================

def plot_scatter(df, x_col, y_col, title, output_path, xlabel, figsize=(10, 6)):
    """Scatter plot of a metric against an ordinal demographic, with trend line."""
    plot_data = df[[x_col, y_col]].dropna()
    if len(plot_data) == 0:
        print(f"Warning: No valid data for {title}")
        return

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(plot_data[x_col], plot_data[y_col], alpha=0.5, s=60,
               color=TUM_BLUE, edgecolors=TUM_BLUE_DARK, linewidth=0.5)

    if len(plot_data) >= 3:
        trend = np.poly1d(np.polyfit(plot_data[x_col], plot_data[y_col], 1))
        x_line = np.linspace(plot_data[x_col].min(), plot_data[x_col].max(), 100)
        ax.plot(x_line, trend(x_line), color=TUM_ORANGE, linewidth=2.5, linestyle="-", label="Trend line")

    _annotate_spearman(ax, plot_data[x_col], plot_data[y_col], n_label="n")

    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    ax.set_ylabel(y_col.replace("_", " "), fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.2, linestyle="--")
    if len(plot_data) >= 3:
        ax.legend(loc="upper left", fontsize=10)

    _save(output_path)


def plot_group_box(df, group_col, y_col, title, output_path, xlabel, order=None,
                   numeric_x=False, figsize=(10, 6), box_width=0.5, jitter_sd=0.08,
                   annotate_spearman=False, xlim=None, xticks=None, xticklabels=None):
    """
    Box plot of a metric per demographic group, with the individual participants
    jittered on top, a mean line and per-group sample sizes.

    With `numeric_x` the groups are placed at their own numeric value (ordinal
    scales such as education level); otherwise they are placed side by side.
    """
    plot_data = df[[group_col, y_col]].dropna()
    if len(plot_data) == 0:
        print(f"Warning: No valid data for {title}")
        return

    present = plot_data[group_col].unique()
    groups = [g for g in order if g in present] if order is not None else sorted(present)
    positions = list(groups) if numeric_x else list(range(len(groups)))
    by_group = [plot_data[plot_data[group_col] == group][y_col].values for group in groups]

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(by_group, positions=positions, widths=box_width, patch_artist=True, **BOX_STYLE)

    for position, values in zip(positions, by_group):
        jitter = np.random.normal(0, jitter_sd, len(values))
        ax.scatter(position + jitter, values, alpha=0.4, s=40, color=TUM_BLUE, edgecolors="none")

    ax.plot(positions, [values.mean() for values in by_group], color=TUM_GREEN, linewidth=2.5,
            marker="D", markersize=8, label="Mean", linestyle="--", zorder=10)

    for position, values in zip(positions, by_group):
        ax.text(position, ax.get_ylim()[1] * 0.98, f"n={len(values)}",
                ha="center", va="top", fontsize=9, color=TUM_GRAY_80)

    if annotate_spearman:
        _annotate_spearman(ax, plot_data[group_col], plot_data[y_col])

    ax.set_xlabel(xlabel, fontsize=13, fontweight="bold")
    ax.set_ylabel(y_col.replace("_", " "), fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=20)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xticks(xticks if xticks is not None else positions)
    ax.set_xticklabels(xticklabels if xticklabels is not None else groups)
    ax.grid(True, alpha=0.2, linestyle="--", axis="y")
    ax.legend(loc="upper left", fontsize=10)

    _save(output_path)


def plot_metric_grid(participant_df, group_col, output_path, suptitle, xlabel, kind="violin",
                     order=None, numeric_x=True, figsize=(18, 10), width=0.6, jitter_sd=0.06,
                     trend_range=None, annotate_rho=False, xlim=None, xticks=None,
                     xticklabels=None, xticklabel_fontsize=None):
    """
    One panel per metric (5 panels in a 2x3 grid) showing its distribution
    across the demographic, either as violins or as a scatter with a trend line.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for ax, metric, label, color in zip(axes, METRICS, METRIC_LABELS, METRIC_COLORS):
        plot_data = participant_df[[group_col, metric]].dropna()

        if len(plot_data) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        if kind == "scatter":
            ax.scatter(plot_data[group_col], plot_data[metric], alpha=0.5, s=50,
                       color=color, edgecolors="black", linewidth=0.5)
        else:
            present = plot_data[group_col].unique()
            groups = [g for g in order if g in present] if order is not None else sorted(present)
            positions = list(groups) if numeric_x else list(range(len(groups)))
            by_group = [plot_data[plot_data[group_col] == group][metric].values for group in groups]

            parts = ax.violinplot(by_group, positions=positions, widths=width,
                                  showmeans=True, showmedians=True, showextrema=True)
            for body in parts["bodies"]:
                body.set_facecolor(color)
                body.set_alpha(0.6)
                body.set_edgecolor(TUM_BLUE_DARK)
            parts["cmeans"].set_color(TUM_GREEN)
            parts["cmeans"].set_linewidth(2)
            parts["cmedians"].set_color(TUM_ORANGE)
            parts["cmedians"].set_linewidth(2)

            for position, values in zip(positions, by_group):
                jitter = np.random.normal(0, jitter_sd, len(values))
                ax.scatter(position + jitter, values, alpha=0.3, s=25, color="black", edgecolors="none")

            if xticks is None:
                ax.set_xticks(positions)
                ax.set_xticklabels(groups, **({"fontsize": xticklabel_fontsize} if xticklabel_fontsize else {}))

        if trend_range is not None and len(plot_data) >= 3:
            trend = np.poly1d(np.polyfit(plot_data[group_col], plot_data[metric], 1))
            span = ((plot_data[group_col].min(), plot_data[group_col].max())
                    if trend_range == "data" else trend_range)
            x_line = np.linspace(*span, 100)
            ax.plot(x_line, trend(x_line), color="red", linewidth=2, linestyle="--", alpha=0.7, label="Trend")

        if annotate_rho:
            rho, p_value = stats.spearmanr(plot_data[group_col], plot_data[metric])
            ax.text(0.02, 0.98, f"ρ = {rho:.3f} {sig_stars(p_value, ns='ns')}", transform=ax.transAxes,
                    fontsize=10, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
        ax.set_ylabel(label, fontsize=11, fontweight="bold")
        ax.set_title(label, fontsize=13, fontweight="bold")
        if xlim is not None:
            ax.set_xlim(*xlim)
        if xticks is not None:
            ax.set_xticks(xticks)
            if xticklabels is not None:
                ax.set_xticklabels(xticklabels, **({"fontsize": xticklabel_fontsize} if xticklabel_fontsize else {}))
        ax.grid(True, alpha=0.2, linestyle="--", axis="y" if kind == "violin" else "both")

    fig.delaxes(axes[5])
    plt.suptitle(suptitle, fontsize=16, fontweight="bold", y=1.00)
    _save(output_path)


def aor_summary_by_group(long_df, group_col):
    """
    Mean RAIR, RSR and AOR per demographic group.

    RAIR is measured on trials where the AI was right and the participant was
    initially wrong, RSR where the AI was wrong and the participant right.
    """
    df_rair = long_df[(long_df["ai_correct"] == 1) & (long_df["human_pre_correct"] == 0)].copy()
    df_rair["rair"] = df_rair["changed_to_correct"].astype(float)
    rair = df_rair.groupby(group_col)["rair"].mean().rename("RAIR")

    df_rsr = long_df[(long_df["ai_correct"] == 0) & (long_df["human_pre_correct"] == 1)].copy()
    df_rsr["rsr"] = df_rsr["stayed_correct"].astype(float)
    rsr = df_rsr.groupby(group_col)["rsr"].mean().rename("RSR")

    summary = pd.concat([rair, rsr], axis=1).reset_index().dropna()
    summary["AOR"] = (summary["RAIR"] + summary["RSR"]) / 2.0
    return summary


def plot_aor_by_group(long_df, group_col, output_path, group_title, legend_title, color_of,
                      label_of=str, order=None, point_size=300, edge_width=2, annotation_fontsize=10,
                      box_pad=0.5, box_linewidth=2, explanation_fontsize=9, legend_markersize=10,
                      legend_fontsize=None, offsets=None, summary_title=None, summary_header="Group",
                      summary_label_of=None, summary_width=15):
    """
    RAIR (x) against RSR (y), one point per demographic group, annotated with
    the group's AOR = (RAIR + RSR) / 2.
    """
    summary = aor_summary_by_group(long_df, group_col)

    if order is not None:
        summary[group_col] = pd.Categorical(summary[group_col], categories=order, ordered=True)
        summary = summary.sort_values(group_col).dropna(subset=[group_col])

    if len(summary) == 0:
        print("Warning: No valid data for AOR plot")
        return

    if offsets is None:
        offsets = lambda i, row: ((10, 10), "left", "bottom")

    colors = [color_of(group) for group in summary[group_col]]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(summary["RAIR"], summary["RSR"], s=point_size, c=colors,
               edgecolor="black", linewidth=edge_width, zorder=10, alpha=0.8)

    for i, (_, row) in enumerate(summary.iterrows()):
        xytext, ha, va = offsets(i, row)
        ax.annotate(
            f"{label_of(row[group_col])}\nAOR={row['AOR']:.3f}",
            (row["RAIR"], row["RSR"]), xytext=xytext, textcoords="offset points",
            ha=ha, va=va, fontsize=annotation_fontsize, fontweight="bold",
            bbox=dict(boxstyle=f"round,pad={box_pad}", facecolor="white",
                      edgecolor=color_of(row[group_col]), alpha=0.95, linewidth=box_linewidth),
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1.5, label="RAIR = RSR")

    ax.set_xlabel("RAIR (Reliance on AI when Right)", fontsize=13, fontweight="bold")
    ax.set_ylabel("RSR (Resistance to wrong AI)", fontsize=13, fontweight="bold")
    ax.set_title(f"Appropriateness of Reliance (AOR) by {group_title}\nRAIR vs RSR",
                 fontsize=15, fontweight="bold", pad=20)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_aspect("equal")

    ax.text(0.02, 0.98, "AOR = (RAIR + RSR) / 2\nHigher values = Better reliance",
            transform=ax.transAxes, fontsize=explanation_fontsize, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor=TUM_BEIGE, alpha=0.8))

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label=label_of(group),
               markerfacecolor=color_of(group), markeredgecolor="black", markersize=legend_markersize)
        for group in summary[group_col]
    ]
    legend_kwargs = {"fontsize": legend_fontsize} if legend_fontsize else {}
    ax.legend(handles=legend_elements, loc="lower right", title=legend_title,
              framealpha=0.95, edgecolor="black", **legend_kwargs)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    summary_label_of = summary_label_of or label_of

    print("\n" + "=" * 80)
    print(f"AOR (Appropriateness of Reliance) by {summary_title or group_title}")
    print("=" * 80)
    print(f"{summary_header:<{summary_width}} {'RAIR':<10} {'RSR':<10} {'AOR':<10}")
    print("-" * 80)
    for _, row in summary.iterrows():
        print(f"{summary_label_of(row[group_col]):<{summary_width}} {row['RAIR']:<10.3f} "
              f"{row['RSR']:<10.3f} {row['AOR']:<10.3f}")
    print("=" * 80)

    print(f"\nSaved AOR plot to: {output_path}")


# ============================================================================
# REGRESSIONS
# ============================================================================

# The five models every demographic script runs, in order.
# (key, heading, estimator, outcome, eligibility subset)
MODEL_SPECS = [
    ("plausibility", "1. PLAUSIBILITY ~ {label} (OLS with cluster-robust SEs)", "ols", "plaus", None),
    ("confidence_change", "2. CONFIDENCE_CHANGE ~ {label} (OLS with cluster-robust SEs)", "ols", "delta_conf", None),
    ("final_accuracy", "3. FINAL_ACCURACY (post==gt) ~ {label} (Logistic with cluster-robust SEs)",
     "logit", "final_correct", None),
    ("rair", "4. RAIR (changed_to_correct | AI correct & human initially wrong)\n"
             "   Logistic regression with cluster-robust SEs", "logit", "changed_to_correct", "rair"),
    ("rsr", "5. RSR (stayed_correct | AI wrong & human initially correct)\n"
            "   Logistic regression with cluster-robust SEs", "logit", "stayed_correct", "rsr"),
]

MIN_ELIGIBLE_TRIALS = 10

# Wording shared by the ordinal and the binary interpretation helpers
_EVENTS = {
    "final_accuracy": "being correct",
    "rair": "accepting AI's correct suggestion",
    "rsr": "resisting AI's wrong suggestion",
}


def ordinal_interpreters(change_phrase):
    """
    Interpretation lines for an ordinal predictor, e.g.
    change_phrase="Each 1-level increase in education".
    """
    outcomes = {"plausibility": "plausibility rating", "confidence_change": "confidence change"}

    def interpret(key, coef, odds_ratio):
        if key in outcomes:
            direction = "increase" if coef > 0 else "decrease"
            return [f"Interpretation: {change_phrase} is associated with",
                    f"               {abs(coef):.4f} {direction} in {outcomes[key]}"]
        return [f"Interpretation: {change_phrase} multiplies the odds",
                f"               of {_EVENTS[key]} by {odds_ratio:.4f}"]

    return interpret


def group_interpreters(subject, reference):
    """
    Interpretation lines for a binary predictor, e.g.
    subject="Females", reference="males".
    """
    outcomes = {"plausibility": "plausibility ratings", "confidence_change": "confidence change"}

    def interpret(key, coef, odds_ratio):
        if key in outcomes:
            direction = "higher" if coef > 0 else "lower"
            return [f"Interpretation: {subject} have {abs(coef):.4f} {direction} "
                    f"{outcomes[key]} than {reference}"]
        suffix = f" compared to {reference}" if key == "final_accuracy" else ""
        return [f"Interpretation: {subject} have {odds_ratio:.4f}x the odds of {_EVENTS[key]}{suffix}"]

    return interpret


def _fit_and_report(key, spec_kind, outcome, data, predictor, interpret):
    """Fit one model, print its summary and interpretation, and return it."""
    fit = ols_clustered if spec_kind == "ols" else logit_clustered
    model = fit(f"{outcome} ~ {predictor}", data=data, cluster_var="participant")
    print(model.summary())

    coef = model.params[predictor]
    stars = sig_stars(model.pvalues[predictor])

    if spec_kind == "ols":
        odds_ratio = None
        print(f"\nCoefficient: {coef:.4f} {stars}")
    else:
        odds_ratio = np.exp(coef)
        print(f"\nLog-odds coefficient: {coef:.4f} {stars}")
        print(f"Odds ratio: {odds_ratio:.4f}")

    for line in interpret(key, coef, odds_ratio):
        print(line)

    return model


def run_clustered_regressions(long_df, predictor, predictor_label, interpret,
                              coerce_numeric=False, coding_note=None):
    """
    Run the five standard regressions (plausibility, confidence change, final
    accuracy, RAIR, RSR) with participant-clustered standard errors.

    Returns a dict mapping each model key to the fitted model (None on failure).
    """
    analysis_df = long_df.copy()
    if coerce_numeric:
        analysis_df[predictor] = pd.to_numeric(analysis_df[predictor], errors="coerce")
    analysis_df["final_correct"] = (analysis_df["post"] == analysis_df["gt"]).astype(int)

    print("\n" + "=" * 80)
    print("REGRESSION ANALYSES WITH CLUSTER-ROBUST STANDARD ERRORS")
    print("=" * 80)
    print("Clustering by participant to account for repeated measures")
    print("Same methodology as h_tests.py")
    if coding_note:
        print(coding_note)

    subsets = {
        "rair": (analysis_df["ai_correct"] == 1) & (analysis_df["human_pre_correct"] == 0),
        "rsr": (analysis_df["ai_correct"] == 0) & (analysis_df["human_pre_correct"] == 1),
    }

    results = {}
    for key, heading, spec_kind, outcome, subset in MODEL_SPECS:
        print("\n" + "-" * 80)
        print(heading.format(label=predictor_label))
        print("-" * 80)

        results[key] = None
        try:
            data = analysis_df
            if subset is not None:
                data = analysis_df[subsets[subset]].copy()
                print(f"{subset.upper()}-eligible trials: {len(data)}")
                if len(data) < MIN_ELIGIBLE_TRIALS:
                    print(f"Insufficient {subset.upper()}-eligible observations for regression")
                    continue

            results[key] = _fit_and_report(key, spec_kind, outcome, data, predictor, interpret)
        except Exception as exc:
            print(f"Error: {exc}")

    print("\n" + "=" * 80)

    return results


def print_completion_summary(output_folder, sections, generated_files):
    """
    Print the closing block shared by all demographic analyses.

    `sections` is a list of (heading, lines) pairs, `generated_files` a list of
    (filename, description) pairs.
    """
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll files saved in folder: {output_folder}/")

    for heading, lines in sections:
        print(f"\n{heading}:")
        for line in lines:
            print(line)

    print("\nGenerated files:")
    for name, description in generated_files:
        print(f"  - {output_folder}/{name} ({description})")

    print("\nRegression results are printed above (full statsmodels output)")
    print("\n")
