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

# Run
if __name__ == "__main__":
    df_trials = build_full_with_deltas_and_labels()

    # RAIR returns 3 values
    df_with_rair, rair_value, rair_counts = compute_rair_global(df_trials, n_trials=16)

    # Feed the RAIR-augmented DF into RSR so you keep both columns
    df_with_rair_rsr, rsr_value, rsr_counts = compute_rsr_global(df_with_rair, n_trials=16)

    print("\n=== DataFrame head with RAIR & RSR columns ===")
    print(df_with_rair_rsr.head().to_string())

    print("\n=== Metric values ===")
    print(f"RAIR: {rair_value!r}")
    print(f"RSR : {rsr_value!r}")

    print("\n=== RAIR counts ===")
    print(rair_counts)

    print("\n=== RSR counts ===")
    print(rsr_counts)