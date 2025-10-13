import pandas as pd

# Load the data
df = pd.read_excel("experiment_results_with_metrics.xlsx")

# Total number of participants
N = len(df)

# Field categorization function (defined early for reuse)
def categorize_field(field):
    if pd.isna(field):
        return 'Other / Unspecified'
    
    field_lower = str(field).lower()
    
    # Computer Science / AI / ML
    if any(keyword in field_lower for keyword in ['computer science', 'ai', 'machine learning', 'ml', 'artificial intelligence']):
        return 'Computer Science / AI / ML'
    
    # Healthcare
    if any(keyword in field_lower for keyword in ['healthcare', 'medicine', 'medical', 'health', 'genetic', 'biology']):
        return 'Healthcare'
    
    # Finance
    if any(keyword in field_lower for keyword in ['finance', 'banking', 'economics']):
        return 'Finance'
    
    # Marketing
    if any(keyword in field_lower for keyword in ['marketing', 'advertising']):
        return 'Marketing'
    
    # Education
    if any(keyword in field_lower for keyword in ['education', 'teaching', 'pedagogy']):
        return 'Education'
    
    # Otherwise, it's Other
    return 'Other / Unspecified'

print(f"Total Participants (N): {N}")
print("\n" + "="*80)
print("DEMOGRAPHICS TABLE - LaTeX FORMAT")
print("="*80 + "\n")

# Age distribution
print("AGE:")
age_counts = df['What is your age?'].value_counts().sort_index()
for age_group in age_counts.index:
    count = age_counts[age_group]
    percent = (count / N) * 100
    print(f"\\quad {age_group:<10} & {count:2d} & {percent:4.1f}\\% \\\\")

print("\nGENDER:")
gender_counts = df['What is your gender?'].value_counts()
for gender in gender_counts.index:
    count = gender_counts[gender]
    percent = (count / N) * 100
    print(f"\\quad {gender:<20} & {count:2d} & {percent:4.1f}\\% \\\\")

print("\nHIGHEST ACHIEVED EDUCATION:")
edu_counts = df['What is your highest achieved level of education?'].value_counts()
for edu in edu_counts.index:
    count = edu_counts[edu]
    percent = (count / N) * 100
    print(f"\\quad {edu:<21} & {count:2d} & {percent:4.1f}\\% \\\\")

print("\nFIELD OF WORK OR STUDY (Raw Data):")
field_counts = df['What is your field of work/study ?'].value_counts()
for field in field_counts.index:
    count = field_counts[field]
    percent = (count / N) * 100
    print(f"  {field:<40} -> {count:2d} ({percent:4.1f}%)")

print("\nFIELD OF WORK OR STUDY (Categorized for LaTeX table):")
# Show categorized version
for field in df['What is your field of work/study ?'].unique():
    category = categorize_field(field)
    print(f"  '{field}' -> {category}")

print("\nEXPERIENCE WITH NLP (Self-Rating, 1-5):")
nlp_exp_counts = df['Please rate your experience with NLP'].value_counts().sort_index()
for rating in nlp_exp_counts.index:
    count = nlp_exp_counts[rating]
    percent = (count / N) * 100
    print(f"\\quad ({int(rating)}) & {count:2d} & {percent:4.1f}\\% \\\\")

print("\n" + "="*80)
print("COMPLETE LaTeX TABLE:")
print("="*80 + "\n")

# Now generate the complete formatted table
print(r"""\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{8pt}
\begin{tabular}{lrr}
\toprule
\textbf{Characteristic} & \textbf{Count} & \textbf{Percent} \\
\midrule
\multicolumn{3}{l}{\textit{Age}} \\""")

# Age - in the order specified in the template
age_order = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
for age_group in age_order:
    if age_group in age_counts.index:
        count = age_counts[age_group]
        percent = (count / N) * 100
        print(f"\\quad {age_group:<8} & {count:2d} & {percent:4.1f}\\% \\\\")
    else:
        print(f"\\quad {age_group:<8} & {0:2d} & {0:4.1f}\\% \\\\")

print(r"""\addlinespace
\multicolumn{3}{l}{\textit{Gender}} \\""")

# Gender - in specified order
gender_order = ['Female', 'Male', 'Prefer not to say']
for gender in gender_order:
    if gender in gender_counts.index:
        count = gender_counts[gender]
        percent = (count / N) * 100
        print(f"\\quad {gender:<19} & {count:2d} & {percent:4.1f}\\% \\\\")
    else:
        print(f"\\quad {gender:<19} & {0:2d} & {0:4.1f}\\% \\\\")

print(r"""\addlinespace
\multicolumn{3}{l}{\textit{Highest Achieved Education}} \\""")

# Education - in specified order (case-insensitive matching)
edu_order = ['High school degree', "Bachelor's degree", "Master's degree", 'PhD or equivalent']
edu_counts_normalized = {}
for edu, count in edu_counts.items():
    edu_lower = str(edu).lower()
    if 'phd' in edu_lower or 'doctorate' in edu_lower:
        edu_counts_normalized['PhD or equivalent'] = edu_counts_normalized.get('PhD or equivalent', 0) + count
    elif 'master' in edu_lower:
        edu_counts_normalized["Master's degree"] = edu_counts_normalized.get("Master's degree", 0) + count
    elif 'bachelor' in edu_lower:
        edu_counts_normalized["Bachelor's degree"] = edu_counts_normalized.get("Bachelor's degree", 0) + count
    elif 'high school' in edu_lower:
        edu_counts_normalized['High school degree'] = edu_counts_normalized.get('High school degree', 0) + count
    else:
        edu_counts_normalized[edu] = count

for edu in edu_order:
    count = edu_counts_normalized.get(edu, 0)
    percent = (count / N) * 100
    print(f"\\quad {edu:<21} & {count:2d} & {percent:4.1f}\\% \\\\")

print(r"""\addlinespace
\multicolumn{3}{l}{\textit{Field of Work or Study}} \\""")

# Create categorized counts (using function defined at top of file)
field_categories = {}
for field in df['What is your field of work/study ?']:
    category = categorize_field(field)
    field_categories[category] = field_categories.get(category, 0) + 1

field_order = ['Computer Science / AI / ML', 'Healthcare', 'Finance', 'Marketing', 'Education', 'Other / Unspecified']
for field in field_order:
    count = field_categories.get(field, 0)
    percent = (count / N) * 100
    print(f"\\quad {field:<28} & {count:2d} & {percent:4.1f}\\% \\\\")

print(r"""\addlinespace
\multicolumn{3}{l}{\textit{Experience with NLP (Self-Rating, 1--5)}} \\""")

# NLP Experience - 1 to 5
for rating in range(1, 6):
    if rating in nlp_exp_counts.index:
        count = nlp_exp_counts[rating]
        percent = (count / N) * 100
        print(f"\\quad ({rating}) & {count:2d} & {percent:4.1f}\\% \\\\")
    else:
        print(f"\\quad ({rating}) & {0:2d} & {0:4.1f}\\% \\\\")

print(r"""\bottomrule
\end{tabular}""")
print(f"\\caption{{Participant demographics for the analyzed sample ($N={N}$). Percentages are relative to the final analyzed $N$. Minor totals may not sum to 100\\% due to rounding or missing responses.}}")
print(r"""\label{tab:demographics}
\end{table*}""")

