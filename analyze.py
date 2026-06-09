"""
analyze.py — Main analysis script for the Gemma MCQ Evaluation Project (02445)

Usage:
    python analyze.py                       # uses data/results.csv
    python analyze.py data/example.csv      # use example data to test the pipeline
    python analyze.py data/results.csv --no-plots
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.loader import load_results, validate_data
from src.tests import (
    overall_accuracy_test,
    mcnemar_modality_test,
    type_comparison_test,
    topic_accuracy,
)
from src.plots import save_all_plots

FIGURES_DIR = Path("figures")
DEFAULT_DATA = Path("data/results.csv")
FALLBACK_DATA = Path("data/example.csv")


# ──────────────────────────────────────────────────────────────
#  Formatting helpers
# ──────────────────────────────────────────────────────────────

W = 64

def _sep(char='─'): print(char * W)
def _head(title): _sep('═'); print(f'  {title}'); _sep('═')
def _sec(title): print(); _sep(); print(f'  {title}'); _sep()


# ──────────────────────────────────────────────────────────────
#  Report sections
# ──────────────────────────────────────────────────────────────

def print_data_summary(df):
    _sec("DATA SUMMARY")
    exam_years = sorted(df['exam_year'].unique())
    n_q = df.groupby(['exam_year', 'question_id']).ngroups
    type_counts = (
        df.drop_duplicates(['exam_year', 'question_id'])['question_type']
        .value_counts().sort_index().to_dict()
    )
    print(f"  Exam sets     : {len(exam_years)}  ({', '.join(exam_years)})")
    print(f"  Total rows    : {len(df)}  (questions × modalities)")
    print(f"  Unique questions: {n_q}")
    print(f"  By type       : " + "  ".join(f"Type {t}: {n}" for t, n in type_counts.items()))
    print(f"  Modalities    : {sorted(df['modality'].unique())}")
    print(f"  Topics        : {sorted(df['topic'].unique())}")


def print_overall_accuracy(df):
    _sec("1. OVERALL ACCURACY (binomial test vs. 25% chance)")
    res = overall_accuracy_test(df)
    for line in res.interpretation.split('\n'):
        print(f"  {line}")


def print_type_comparison(df):
    _sec("2. ACCURACY BY QUESTION TYPE (chi-square)")
    res = type_comparison_test(df)
    acc = res.details['accuracy_by_type']
    for qtype in sorted(acc.keys()):
        v = acc[qtype]
        ci = v['ci']
        desc = {'A': 'Pure text    ', 'B': 'Figure/graph ', 'C': 'Table/matrix '}
        print(f"  Type {qtype} ({desc.get(qtype, '')}): "
              f"{v['accuracy']:.1%}  "
              f"[{ci[0]:.1%}, {ci[1]:.1%}] 95% CI  "
              f"(n={v['n']})")
    print()
    for line in res.interpretation.split('\n'):
        print(f"  {line}")


def print_mcnemar(df):
    _sec("3. MODALITY COMPARISON — McNemar's Test")
    print("  H0: P(correct | screenshot) = P(correct | text description)")
    print()
    results = mcnemar_modality_test(df)
    if not results:
        print("  ✗ Not enough paired observations. Fill in both modalities for Type B/C questions.")
        return
    for label, res in results.items():
        print(f"  ── Type {label} ──────────────────────────────────")
        for line in res.interpretation.split('\n'):
            print(f"  {line}")
        print()


def print_topics(df):
    _sec("4. ACCURACY BY TOPIC (descriptive)")
    topics = topic_accuracy(df)
    bar_max = 30
    for topic, v in topics.items():
        bar_len = int(v['accuracy'] * bar_max)
        bar = '█' * bar_len + '░' * (bar_max - bar_len)
        chance_pos = int(0.25 * bar_max)
        bar = bar[:chance_pos] + '|' + bar[chance_pos + 1:]
        ci = v['ci']
        print(f"  {topic:<25} {bar}  {v['accuracy']:.0%}  "
              f"[{ci[0]:.0%}, {ci[1]:.0%}]  n={v['n']}")
    print()
    print("  (| marks chance level 25%)")


def print_assumptions(df):
    _sec("5. ASSUMPTION CHECKS")

    # McNemar: enough discordant pairs?
    results = mcnemar_modality_test(df)
    if results:
        for label, res in results.items():
            disc = res.details['discordant']
            method = 'exact' if res.details['exact'] else 'asymptotic'
            status = '✓' if disc >= 5 else '⚠ low (<5)'
            print(f"  McNemar Type {label}: discordant pairs = {disc}  → {method} test  {status}")

    # Chi-square: minimum expected cell count
    from src.loader import primary_observations
    from scipy.stats import chi2_contingency
    import pandas as pd
    primary = primary_observations(df)
    contingency = pd.crosstab(primary['question_type'], primary['is_correct'])
    for col in [0, 1]:
        if col not in contingency.columns:
            contingency[col] = 0
    _, _, _, expected = chi2_contingency(contingency[[0, 1]])
    min_exp = expected.min()
    status = '✓' if min_exp >= 5 else f'⚠ min expected = {min_exp:.1f} (use Fisher\'s)'
    print(f"  Chi-square (type comparison): min expected cell = {min_exp:.1f}  {status}")

    # Independence note
    print()
    print("  Note: independence assumption — each question was asked in a fresh")
    print("  Gemma session (new conversation). Please verify this in your data.")


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────

def main():
    # Argument parsing
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = [a for a in sys.argv[1:] if a.startswith('--')]
    skip_plots = '--no-plots' in flags

    data_path = Path(args[0]) if args else DEFAULT_DATA
    if not data_path.exists():
        if FALLBACK_DATA.exists():
            print(f"  '{data_path}' not found — using example data instead.")
            data_path = FALLBACK_DATA
        else:
            print(f"  Error: '{data_path}' not found.")
            print(f"  Run:  python analyze.py data/example.csv   to try the example.")
            sys.exit(1)

    _head("GEMMA MCQ EVALUATION — STATISTICAL REPORT (02445)")

    df = load_results(data_path)
    validate_data(df)

    print_data_summary(df)
    print_overall_accuracy(df)
    print_type_comparison(df)
    print_mcnemar(df)
    print_topics(df)
    print_assumptions(df)

    if not skip_plots:
        _sec("6. FIGURES")
        save_all_plots(df, FIGURES_DIR)
        print(f"\n  Figures saved to: {FIGURES_DIR}/")

    _sep('═')
    print("  Done. Copy test statistics into your report.")
    _sep('═')
    print()


if __name__ == '__main__':
    main()
