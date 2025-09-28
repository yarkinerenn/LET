import pandas as pd
import math

def normalize_series_DT(series: pd.Series) -> pd.Series:
    return (series.astype(str)
                 .str.strip()
                 .str.lower()
                 .map({"d":"D","deceptive":"D","t":"T","truthful":"T"})
                 .fillna(""))

def compute_rsr_global(df_trials: pd.DataFrame, n_trials: int = 16) -> tuple[pd.DataFrame, float, dict]:
    """
    RSR = stayed_correct / (stayed_correct + switched_to_wrong)
    Restricted to cases where:
      - Human initially correct (Pre == GT)
      - AI wrong (AI != GT)
    Uses Pre = Qn_Review, Post = Qn_ReviewExp.
    Appends a constant 'RSR' column to df_trials and returns (df_with_rsr, rsr_value, counts_dict).
    """
    stayed_correct = 0
    switched_wrong = 0
    eligible_total = 0

    for q in range(1, n_trials + 1):
        need = [f"Q{q}_Review", f"Q{q}_ReviewExp", f"Q{q}_GT", f"Q{q}_AI"]
        if not all(c in df_trials.columns for c in need):
            continue

        pre  = normalize_series_DT(df_trials[f"Q{q}_Review"])
        post = normalize_series_DT(df_trials[f"Q{q}_ReviewExp"])
        gt   = df_trials[f"Q{q}_GT"].astype(str).str.strip().str.upper()
        ai   = df_trials[f"Q{q}_AI"].astype(str).str.strip().str.upper()

        valid = pre.isin(["D","T"]) & post.isin(["D","T"]) & gt.isin(["D","T"]) & ai.isin(["D","T"])
        mask  = valid & (pre == gt) & (ai != gt)   # human initially correct, AI wrong

        eligible_total += int(mask.sum())
        stayed_correct += int((mask & (post == gt)).sum())   # human sticks with correct answer
        switched_wrong += int((mask & (post != gt)).sum())   # human follows wrong AI

    denom = stayed_correct + switched_wrong
    if denom == 0:
        rsr = math.nan  # no eligible cases
    else:
        rsr = stayed_correct / denom  # can be 0.0

    out = df_trials.copy()
    out["RSR"] = rsr

    print("=== RSR Summary ===")
    print(f"Eligible (Human initially correct & AI wrong): {eligible_total}")
    print(f"Stayed correct (ignored wrong AI): {stayed_correct}")
    print(f"Switched to wrong (followed AI): {switched_wrong}")
    if math.isnan(rsr):
        print("RSR = NaN (no eligible cases)")
    else:
        print(f"RSR = {rsr:.4f}")

    return out, rsr, {
        "eligible_total": eligible_total,
        "stayed_correct": stayed_correct,
        "switched_wrong": switched_wrong
    }

def compute_rair_global(df_trials: pd.DataFrame, n_trials: int = 16) -> tuple[pd.DataFrame, float, dict]:
    """
    RAIR = corrected / (corrected + not_corrected)
    Only over cases where:
      - AI is correct (AI == GT)
      - Human initially wrong (Pre != GT)
    Pre = Qn_Review, Post = Qn_ReviewExp.
    """
    corrected = 0
    not_corrected = 0
    eligible_total = 0

    for q in range(1, n_trials + 1):
        need = [f"Q{q}_Review", f"Q{q}_ReviewExp", f"Q{q}_GT", f"Q{q}_AI"]
        if not all(c in df_trials.columns for c in need):
            continue

        pre  = normalize_series_DT(df_trials[f"Q{q}_Review"])
        post = normalize_series_DT(df_trials[f"Q{q}_ReviewExp"])
        gt   = df_trials[f"Q{q}_GT"].astype(str).str.strip().str.upper()
        ai   = df_trials[f"Q{q}_AI"].astype(str).str.strip().str.upper()

        valid = pre.isin(["D","T"]) & post.isin(["D","T"]) & gt.isin(["D","T"]) & ai.isin(["D","T"])
        mask  = valid & (ai == gt) & (pre != gt)

        eligible_total += int(mask.sum())
        corrected      += int((mask & (post == gt)).sum())
        not_corrected  += int((mask & (post != gt)).sum())

    denom = corrected + not_corrected
    if denom == 0:
        rair = math.nan  # no eligible cases at all
    else:
        rair = corrected / denom  # this can be 0.0, 0.5, 1.0, etc.

    # append RAIR as a column (same value for all rows)
    out = df_trials.copy()
    out["RAIR"] = rair

    print("=== RAIR Summary ===")
    print(f"Eligible (AI correct & human initially wrong): {eligible_total}")
    print(f"Corrected after advice: {corrected}")
    print(f"Not corrected after advice: {not_corrected}")
    if math.isnan(rair):
        print("RAIR = NaN (no eligible cases)")
    else:
        # prints 0.0000 properly when corrected == 0
        print(f"RAIR = {rair:.4f}")

    return out, rair, {
        "eligible_total": eligible_total,
        "corrected": corrected,
        "not_corrected": not_corrected
    }
def build_full_with_deltas_and_labels():
    # === Config ===
    file_path = "experiment.xlsx"
    sheet = 0
    start_col_index = 8   # 0-based index of the first trial column
    n_trials = 16
    block_size = 5

    # === Compact strings (no spaces), length should be n_trials ===
    GT_STRING     = "DTDTDTDTTDTDTDTD"   # Ground truth per question (D/T)
    AI_STRING     = "DTTDDTTDTDDTTDDT"   # AI prediction per question (D/T)
    FAITH_STRING  = "FUFUFUFUFUFUFUFU"   # Faithfulness per question (F/U)

    # === Convert to per-question lists ===
    GT_LABELS    = list(GT_STRING.strip())
    AI_LABELS    = list(AI_STRING.strip())
    FAITH_LABELS = list(FAITH_STRING.strip())

    # === Load ===
    df_orig = pd.read_excel(file_path, sheet_name=sheet)
    original_cols = list(df_orig.columns)

    # === Rename trial columns by position ===
    mapping = {}
    for i in range(n_trials):
        base = start_col_index + i * block_size
        cols = original_cols[base: base + block_size]
        if len(cols) != block_size:
            raise ValueError(f"Trial {i+1}: expected {block_size} columns, got {len(cols)}")
        mapping[cols[0]] = f"Q{i+1}_Review"
        mapping[cols[1]] = f"Q{i+1}_Conf1"
        mapping[cols[2]] = f"Q{i+1}_ReviewExp"
        mapping[cols[3]] = f"Q{i+1}_Conf2"
        mapping[cols[4]] = f"Q{i+1}_Plausibility"

    df = df_orig.rename(columns=mapping)

    # === Compute deltas and drop Conf1/Conf2 ===
    for i in range(1, n_trials + 1):
        c1, c2, d = f"Q{i}_Conf1", f"Q{i}_Conf2", f"Q{i}_Delta"
        if c1 in df.columns and c2 in df.columns:
            conf1 = pd.to_numeric(df[c1], errors="coerce")
            conf2 = pd.to_numeric(df[c2], errors="coerce")
            df[d] = conf2 - conf1
            df.drop(columns=[c1, c2], inplace=True)

    # === Normalizers ===
    def normalize_DT(label: str) -> str:
        if not isinstance(label, str):
            return ""
        s = label.strip().lower()
        if s in {"d", "deceptive"}:  return "D"
        if s in {"t", "truthful"}:   return "T"
        return ""

    def normalize_FU(label: str) -> str:
        if not isinstance(label, str):
            return ""
        s = label.strip().lower()
        if s in {"f", "faithful"}:     return "F"
        if s in {"u", "unfaithful"}:   return "U"
        return ""

    def normalize_series_DT(series: pd.Series) -> pd.Series:
        return (series.astype(str)
                      .str.strip()
                      .str.lower()
                      .map({"d":"D", "deceptive":"D", "t":"T", "truthful":"T"})
                      .fillna(""))

    # === Add GT, AI, Faith columns per question ===
    for i in range(1, n_trials + 1):
        gt    = normalize_DT(GT_LABELS[i-1])    if i-1 < len(GT_LABELS)    else ""
        ai    = normalize_DT(AI_LABELS[i-1])    if i-1 < len(AI_LABELS)    else ""
        faith = normalize_FU(FAITH_LABELS[i-1]) if i-1 < len(FAITH_LABELS) else ""
        df[f"Q{i}_GT"]    = gt
        df[f"Q{i}_AI"]    = ai
        df[f"Q{i}_Faith"] = faith

    # === Disagreement column (human AFTER explanation vs AI) ===
    for i in range(1, n_trials + 1):
        human_col = f"Q{i}_ReviewExp"
        ai_col    = f"Q{i}_AI"
        if human_col in df.columns and ai_col in df.columns:
            human_norm = normalize_series_DT(df[human_col])
            ai_const   = df[ai_col]
            df[f"Q{i}_Disagree"] = (human_norm != ai_const) & human_norm.isin(["D","T"]) & ai_const.isin(["D","T"])
        else:
            df[f"Q{i}_Disagree"] = False

    # === Arrange trial columns ===
    trial_cols = []
    for i in range(1, n_trials + 1):
        for name in (
            f"Q{i}_Review",
            f"Q{i}_ReviewExp",
            f"Q{i}_Plausibility",
            f"Q{i}_Delta",
            f"Q{i}_GT",
            f"Q{i}_AI",
            f"Q{i}_Faith",
            f"Q{i}_Disagree",
        ):
            if name in df.columns:
                trial_cols.append(name)

    df_trials = df[trial_cols]

    # === Preview full trial data ===
    print("=== Trial-only DataFrame head (first 5 rows) ===")
    print(df_trials.head().to_string())

    # === Preview only the disagreement columns ===
    disagree_cols = [c for c in df_trials.columns if c.endswith("_Disagree")]
    print("\n=== Disagreement columns (first 5 rows) ===")
    print(df_trials[disagree_cols].head().to_string())

    return df_trials

import math

def compute_rair_rsr_global_and_per_user(df_trials: pd.DataFrame, n_trials: int = 16) -> tuple[pd.DataFrame, dict]:
    """
    Adds 4 columns:
      - RAIR_global (constant for all rows)
      - RSR_global  (constant for all rows)
      - RAIR_user   (per participant / row)
      - RSR_user    (per participant / row)
    Uses:
      Pre  = Qn_Review
      Post = Qn_ReviewExp
      GT   = Qn_GT
      AI   = Qn_AI
    """
    def _norm_DT(series: pd.Series) -> pd.Series:
        return (series.astype(str)
                     .str.strip()
                     .str.lower()
                     .map({"d":"D","deceptive":"D","t":"T","truthful":"T"})
                     .fillna(""))

    # ---------- GLOBAL ----------
    g_corrected = g_notcorr = g_elig_rair = 0
    g_stayed = g_switched = g_elig_rsr = 0

    for q in range(1, n_trials + 1):
        need = [f"Q{q}_Review", f"Q{q}_ReviewExp", f"Q{q}_GT", f"Q{q}_AI"]
        if not all(c in df_trials.columns for c in need):
            continue

        pre  = _norm_DT(df_trials[f"Q{q}_Review"])
        post = _norm_DT(df_trials[f"Q{q}_ReviewExp"])
        gt   = df_trials[f"Q{q}_GT"].astype(str).str.strip().str.upper()
        ai   = df_trials[f"Q{q}_AI"].astype(str).str.strip().str.upper()

        valid = pre.isin(["D","T"]) & post.isin(["D","T"]) & gt.isin(["D","T"]) & ai.isin(["D","T"]) 

        # RAIR elig: AI correct & human initially wrong
        mask_rair = valid & (ai == gt) & (pre != gt)
        g_elig_rair   += int(mask_rair.sum())
        g_corrected   += int((mask_rair & (post == gt)).sum())
        g_notcorr     += int((mask_rair & (post != gt)).sum())

        # RSR elig: human initially correct & AI wrong
        mask_rsr = valid & (pre == gt) & (ai != gt)
        g_elig_rsr    += int(mask_rsr.sum())
        g_stayed      += int((mask_rsr & (post == gt)).sum())
        g_switched    += int((mask_rsr & (post != gt)).sum())

    rair_global = (g_corrected / (g_corrected + g_notcorr)) if (g_corrected + g_notcorr) > 0 else math.nan
    rsr_global  = (g_stayed / (g_stayed + g_switched))     if (g_stayed + g_switched) > 0 else math.nan

    # ---------- PER-USER (row-wise) ----------
    rair_user = []
    rsr_user  = []

    for _, row in df_trials.iterrows():
        u_corrected = u_notcorr = 0
        u_stayed = u_switched = 0

        for q in range(1, n_trials + 1):
            need = [f"Q{q}_Review", f"Q{q}_ReviewExp", f"Q{q}_GT", f"Q{q}_AI"]
            if not all(c in df_trials.columns for c in need):
                continue

            pre  = str(row.get(f"Q{q}_Review", "")).strip()
            post = str(row.get(f"Q{q}_ReviewExp", "")).strip()
            gt   = str(row.get(f"Q{q}_GT", "")).strip().upper()
            ai   = str(row.get(f"Q{q}_AI", "")).strip().upper()

            # normalize single values
            pre  = {"d":"D","deceptive":"D","t":"T","truthful":"T"}.get(pre.lower(), "")
            post = {"d":"D","deceptive":"D","t":"T","truthful":"T"}.get(post.lower(), "")

            if pre not in {"D","T"} or post not in {"D","T"} or gt not in {"D","T"} or ai not in {"D","T"}:
                continue

            # RAIR per-user
            if ai == gt and pre != gt:
                if post == gt: u_corrected += 1
                else:          u_notcorr  += 1

            # RSR per-user
            if pre == gt and ai != gt:
                if post == gt: u_stayed   += 1
                else:          u_switched += 1

        u_rair = (u_corrected / (u_corrected + u_notcorr)) if (u_corrected + u_notcorr) > 0 else math.nan
        u_rsr  = (u_stayed / (u_stayed + u_switched))     if (u_stayed + u_switched) > 0 else math.nan
        rair_user.append(u_rair)
        rsr_user.append(u_rsr)

    out = df_trials.copy()
    out["RAIR_global"] = rair_global
    out["RSR_global"]  = rsr_global
    out["RAIR_user"]   = rair_user
    out["RSR_user"]    = rsr_user

    # quick print
    print("=== GLOBAL ===")
    print(f"RAIR_global: {('NaN' if math.isnan(rair_global) else f'{rair_global:.4f}')}  "
          f"(eligible={g_elig_rair}, corrected={g_corrected}, not_corrected={g_notcorr})")
    print(f"RSR_global : {('NaN' if math.isnan(rsr_global)  else f'{rsr_global:.4f}')}  "
          f"(eligible={g_elig_rsr}, stayed={g_stayed}, switched={g_switched})")

    print("\n=== PER-USER (first 10) ===")
    print(out[["RAIR_user","RSR_user"]].head(10).to_string(index=False))

    summary = {
        "rair_global": rair_global,
        "rsr_global": rsr_global,
        "global_counts": {
            "rair": {"eligible": g_elig_rair, "corrected": g_corrected, "not_corrected": g_notcorr},
            "rsr":  {"eligible": g_elig_rsr,  "stayed": g_stayed, "switched": g_switched},
        }
    }
    return out, summary

# Run
if __name__ == "__main__":
    df_trials = build_full_with_deltas_and_labels()

    # Compute global + per-user metrics in one go (and append columns)
    df_with_metrics, metrics_summary = compute_rair_rsr_global_and_per_user(df_trials, n_trials=16)

    print("\n=== DataFrame head with global & per-user metrics ===")
    print(df_with_metrics.head().to_string())

    print("\n=== Metrics Summary ===")
    print(metrics_summary)

    # Export DataFrame with metrics to Excel
    output_excel_path = "experiment_results_with_metrics.xlsx"
    df_with_metrics.to_excel(output_excel_path, index=False)
    print(f"DataFrame with metrics exported to '{output_excel_path}'.")