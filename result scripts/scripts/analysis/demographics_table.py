"""
Demographics table
==================

Prints the participant demographics of the analyzed sample, first as raw
counts per question and then as the complete LaTeX table used in the paper.
"""

import pandas as pd

from .. import config

AGE_COLUMN = config.DEMOGRAPHIC_COLUMNS["age"]
GENDER_COLUMN = config.DEMOGRAPHIC_COLUMNS["gender"]
EDUCATION_COLUMN = config.DEMOGRAPHIC_COLUMNS["education"]
JOB_COLUMN = config.DEMOGRAPHIC_COLUMNS["job"]
NLP_COLUMN = config.DEMOGRAPHIC_COLUMNS["nlp_experience"]

# Row order used in the LaTeX table (missing categories are printed as zero)
AGE_ORDER = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
GENDER_ORDER = ["Female", "Male", "Prefer not to say"]
EDUCATION_ORDER = ["High school degree", "Bachelor's degree", "Master's degree", "PhD or equivalent"]
FIELD_ORDER = [
    "Computer Science / AI / ML",
    "Healthcare",
    "Finance",
    "Marketing",
    "Education",
    "Other / Unspecified",
]

# Free-text field of work/study mapped onto the table's categories
FIELD_KEYWORDS = [
    ("Computer Science / AI / ML", ["computer science", "ai", "machine learning", "ml", "artificial intelligence"]),
    ("Healthcare", ["healthcare", "medicine", "medical", "health", "genetic", "biology"]),
    ("Finance", ["finance", "banking", "economics"]),
    ("Marketing", ["marketing", "advertising"]),
    ("Education", ["education", "teaching", "pedagogy"]),
]
OTHER_FIELD = "Other / Unspecified"

# Free-text education answers mapped onto EDUCATION_ORDER
EDUCATION_KEYWORDS = [
    ("PhD or equivalent", ["phd", "doctorate"]),
    ("Master's degree", ["master"]),
    ("Bachelor's degree", ["bachelor"]),
    ("High school degree", ["high school"]),
]


def categorize_field(field):
    """Map a free-text field of work/study onto one of the table categories."""
    if pd.isna(field):
        return OTHER_FIELD

    field_lower = str(field).lower()
    for category, keywords in FIELD_KEYWORDS:
        if any(keyword in field_lower for keyword in keywords):
            return category
    return OTHER_FIELD


def normalize_education_counts(edu_counts):
    """Collapse free-text education answers onto the canonical levels."""
    normalized = {}
    for edu, count in edu_counts.items():
        edu_lower = str(edu).lower()
        level = next(
            (name for name, keywords in EDUCATION_KEYWORDS
             if any(keyword in edu_lower for keyword in keywords)),
            edu,
        )
        normalized[level] = normalized.get(level, 0) + count
    return normalized


def print_raw_counts(df, n_total, counts):
    """Print the per-question counts that back the LaTeX table."""
    print(f"Total Participants (N): {n_total}")
    print("\n" + "=" * 80)
    print("DEMOGRAPHICS TABLE - LaTeX FORMAT")
    print("=" * 80 + "\n")

    print("AGE:")
    for age_group, count in counts["age"].items():
        print(f"\\quad {age_group:<10} & {count:2d} & {count / n_total * 100:4.1f}\\% \\\\")

    print("\nGENDER:")
    for gender, count in counts["gender"].items():
        print(f"\\quad {gender:<20} & {count:2d} & {count / n_total * 100:4.1f}\\% \\\\")

    print("\nHIGHEST ACHIEVED EDUCATION:")
    for edu, count in counts["education"].items():
        print(f"\\quad {edu:<21} & {count:2d} & {count / n_total * 100:4.1f}\\% \\\\")

    print("\nFIELD OF WORK OR STUDY (Raw Data):")
    for field, count in counts["field"].items():
        print(f"  {field:<40} -> {count:2d} ({count / n_total * 100:4.1f}%)")

    print("\nFIELD OF WORK OR STUDY (Categorized for LaTeX table):")
    for field in df[JOB_COLUMN].unique():
        print(f"  '{field}' -> {categorize_field(field)}")

    print("\nEXPERIENCE WITH NLP (Self-Rating, 1-5):")
    for rating, count in counts["nlp"].items():
        print(f"\\quad ({int(rating)}) & {count:2d} & {count / n_total * 100:4.1f}\\% \\\\")


def _print_rows(labels, counts, n_total, width):
    """Print one table block: every label in order, zero-filled when absent."""
    for label in labels:
        count = counts.get(label, 0)
        print(f"\\quad {label:<{width}} & {count:2d} & {count / n_total * 100:4.1f}\\% \\\\")


def print_latex_table(df, n_total, counts):
    """Print the complete LaTeX table."""
    print("\n" + "=" * 80)
    print("COMPLETE LaTeX TABLE:")
    print("=" * 80 + "\n")

    print(r"""\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{8pt}
\begin{tabular}{lrr}
\toprule
\textbf{Characteristic} & \textbf{Count} & \textbf{Percent} \\
\midrule
\multicolumn{3}{l}{\textit{Age}} \\""")
    _print_rows(AGE_ORDER, counts["age"], n_total, width=8)

    print(r"""\addlinespace
\multicolumn{3}{l}{\textit{Gender}} \\""")
    _print_rows(GENDER_ORDER, counts["gender"], n_total, width=19)

    print(r"""\addlinespace
\multicolumn{3}{l}{\textit{Highest Achieved Education}} \\""")
    education = normalize_education_counts(counts["education"])
    _print_rows(EDUCATION_ORDER, education, n_total, width=21)

    print(r"""\addlinespace
\multicolumn{3}{l}{\textit{Field of Work or Study}} \\""")
    fields = df[JOB_COLUMN].apply(categorize_field).value_counts()
    _print_rows(FIELD_ORDER, fields, n_total, width=28)

    print(r"""\addlinespace
\multicolumn{3}{l}{\textit{Experience with NLP (Self-Rating, 1--5)}} \\""")
    for rating in range(1, 6):
        count = counts["nlp"].get(rating, 0)
        print(f"\\quad ({rating}) & {count:2d} & {count / n_total * 100:4.1f}\\% \\\\")

    print(r"""\bottomrule
\end{tabular}""")
    print(
        f"\\caption{{Participant demographics for the analyzed sample ($N={n_total}$). "
        f"Percentages are relative to the final analyzed $N$. Minor totals may not sum "
        f"to 100\\% due to rounding or missing responses.}}"
    )
    print(r"""\label{tab:demographics}
\end{table*}""")


def main():
    """Print the demographics of the processed sample."""
    df = pd.read_excel(str(config.PROCESSED_DATA_FILE))
    n_total = len(df)

    counts = {
        "age": df[AGE_COLUMN].value_counts().sort_index(),
        "gender": df[GENDER_COLUMN].value_counts(),
        "education": df[EDUCATION_COLUMN].value_counts(),
        "field": df[JOB_COLUMN].value_counts(),
        "nlp": df[NLP_COLUMN].value_counts().sort_index(),
    }

    print_raw_counts(df, n_total, counts)
    print_latex_table(df, n_total, counts)


if __name__ == "__main__":
    main()
