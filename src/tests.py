"""
Statistical tests for the Gemma MCQ evaluation project.

Tests implemented:
  1. overall_accuracy_test  — Binomial test vs. chance (0.25)
  2. mcnemar_modality_test  — McNemar's test: screenshot vs. text_desc (paired)
  3. type_comparison_test   — Chi-square: does accuracy differ across types A / B / C?
  4. topic_accuracy         — Accuracy + Wilson CI per topic (descriptive)
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2_contingency, fisher_exact
from statsmodels.stats.contingency_tables import mcnemar as _mcnemar
from statsmodels.stats.proportion import proportion_confint

from .loader import primary_observations


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    statistic: float
    p_value: float
    interpretation: str
    details: dict = field(default_factory=dict)


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson (score) confidence interval for a proportion."""
    return proportion_confint(k, n, alpha=alpha, method='wilson')


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "p < 0.001 ***"
    if p < 0.01:
        return f"p = {p:.4f} **"
    if p < 0.05:
        return f"p = {p:.4f} *"
    return f"p = {p:.4f} (n.s.)"


# ──────────────────────────────────────────────────────────────
#  1. Overall accuracy vs. random chance
# ──────────────────────────────────────────────────────────────

def overall_accuracy_test(df: pd.DataFrame, chance: float = 0.25) -> TestResult:
    """
    One-sided binomial test.
    H0: accuracy = chance (random guessing, 4 options → 0.25)
    H1: accuracy > chance

    Uses primary_observations (one row per question) to avoid double-counting
    questions that were asked in multiple modalities.
    """
    primary = primary_observations(df)
    n = len(primary)
    k = int(primary['is_correct'].sum())
    acc = k / n
    ci = wilson_ci(k, n)

    res = binomtest(k, n, chance, alternative='greater')

    return TestResult(
        name="Overall Accuracy vs. Chance",
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        interpretation=(
            f"Accuracy = {acc:.1%}  (95% CI: [{ci[0]:.1%}, {ci[1]:.1%}])\n"
            f"  n = {n} questions,  correct = {k}\n"
            f"  H0: p = {chance:.0%} (chance),  H1: p > {chance:.0%}\n"
            f"  {fmt_p(res.pvalue)}"
        ),
        details={'n': n, 'correct': k, 'accuracy': acc, 'ci': ci, 'chance': chance},
    )


# ──────────────────────────────────────────────────────────────
#  2. McNemar's test — screenshot vs. text_desc
# ──────────────────────────────────────────────────────────────

def mcnemar_modality_test(df: pd.DataFrame) -> dict[str, TestResult]:
    """
    McNemar's test for paired modality comparison.
    Applied separately to Type B, Type C, and combined B+C.

    For each question that was answered in BOTH modalities:
      a = screenshot correct AND text_desc correct  (concordant ✓)
      b = screenshot correct AND text_desc wrong    (discordant — screenshot wins)
      c = screenshot wrong  AND text_desc correct   (discordant — text_desc wins)
      d = screenshot wrong  AND text_desc wrong     (concordant ✗)

    McNemar's tests H0: b = c (both modalities equally accurate).
    Uses exact binomial when b+c < 25, asymptotic chi-square otherwise.
    """
    results: dict[str, TestResult] = {}

    groups = {
        'B':   df[df['question_type'] == 'B'],
        'C':   df[df['question_type'] == 'C'],
        'B+C': df[df['question_type'].isin(['B', 'C'])],
    }

    for label, subset in groups.items():
        paired = (
            subset[subset['modality'].isin(['screenshot', 'text_desc'])]
            .pivot_table(
                index=['exam_year', 'question_id'],
                columns='modality',
                values='is_correct',
                aggfunc='first',
            )
            .dropna(subset=['screenshot', 'text_desc'])
        )

        if len(paired) < 3:
            continue

        a = int(((paired['screenshot'] == 1) & (paired['text_desc'] == 1)).sum())
        b = int(((paired['screenshot'] == 1) & (paired['text_desc'] == 0)).sum())
        c = int(((paired['screenshot'] == 0) & (paired['text_desc'] == 1)).sum())
        d = int(((paired['screenshot'] == 0) & (paired['text_desc'] == 0)).sum())

        table = np.array([[a, b], [c, d]])
        n = len(paired)
        exact = (b + c) < 25
        res = _mcnemar(table, exact=exact, correction=True)

        acc_ss = (a + b) / n
        acc_td = (a + c) / n
        ci_ss = wilson_ci(a + b, n)
        ci_td = wilson_ci(a + c, n)

        results[label] = TestResult(
            name=f"McNemar's Test — Type {label}: Screenshot vs. Text Description",
            statistic=float(res.statistic),
            p_value=float(res.pvalue),
            interpretation=(
                f"n = {n} paired questions,  discordant pairs (b+c) = {b + c}\n"
                f"  Screenshot : {acc_ss:.1%}  (95% CI: [{ci_ss[0]:.1%}, {ci_ss[1]:.1%}])\n"
                f"  Text desc. : {acc_td:.1%}  (95% CI: [{ci_td[0]:.1%}, {ci_td[1]:.1%}])\n"
                f"  Contingency table: [[a={a}, b={b}], [c={c}, d={d}]]\n"
                f"  Method: {'Exact (binomial)' if exact else 'Asymptotic (χ²)'}\n"
                f"  {fmt_p(res.pvalue)}"
            ),
            details={
                'n_pairs': n,
                'table': table,
                'a': a, 'b': b, 'c': c, 'd': d,
                'acc_screenshot': acc_ss,
                'acc_text_desc': acc_td,
                'ci_screenshot': ci_ss,
                'ci_text_desc': ci_td,
                'discordant': b + c,
                'exact': exact,
            },
        )

    return results


# ──────────────────────────────────────────────────────────────
#  3. Chi-square: accuracy across question types A / B / C
# ──────────────────────────────────────────────────────────────

def type_comparison_test(df: pd.DataFrame) -> TestResult:
    """
    Chi-square test of homogeneity across question types.
    H0: P(correct | type=A) = P(correct | type=B) = P(correct | type=C)

    Uses primary_observations for a fair comparison (one row per question,
    same kind of text-based input for types B and C).
    Falls back to Fisher's exact test when expected cell counts are < 5.
    """
    primary = primary_observations(df)
    contingency = pd.crosstab(primary['question_type'], primary['is_correct'])
    for col in [0, 1]:
        if col not in contingency.columns:
            contingency[col] = 0
    contingency = contingency[[0, 1]]

    _, _, dof, expected = chi2_contingency(contingency)
    use_fisher = expected.min() < 5 and contingency.shape[0] == 2

    if use_fisher:
        stat, p = fisher_exact(contingency.values)
        test_name, dof = "Fisher's Exact", None
    else:
        stat, p, dof, _ = chi2_contingency(contingency, correction=(contingency.shape[0] == 2))
        test_name = "Chi-Square"

    acc_details: dict = {}
    for qtype in sorted(primary['question_type'].unique()):
        sub = primary[primary['question_type'] == qtype]
        k, n = int(sub['is_correct'].sum()), len(sub)
        acc_details[qtype] = {'accuracy': k / n, 'ci': wilson_ci(k, n), 'n': n}

    summary = "  ".join(
        f"{t}: {v['accuracy']:.1%} (n={v['n']})"
        for t, v in sorted(acc_details.items())
    )

    return TestResult(
        name=f"{test_name}: Accuracy Across Question Types",
        statistic=float(stat),
        p_value=float(p),
        interpretation=(
            f"Accuracy — {summary}\n"
            f"  H0: equal accuracy across types A, B, C\n"
            f"  {test_name}: stat = {stat:.3f}"
            + (f", df = {dof}" if dof else "")
            + f"\n  {fmt_p(p)}"
        ),
        details={'accuracy_by_type': acc_details, 'dof': dof, 'contingency': contingency},
    )


# ──────────────────────────────────────────────────────────────
#  4. Topic-level accuracy (descriptive)
# ──────────────────────────────────────────────────────────────

def topic_accuracy(df: pd.DataFrame) -> dict[str, dict]:
    """
    Accuracy + 95% Wilson CI per topic using primary observations.
    Returns dict sorted by accuracy descending.
    """
    primary = primary_observations(df)
    result: dict[str, dict] = {}
    for topic, group in primary.groupby('topic'):
        k, n = int(group['is_correct'].sum()), len(group)
        result[topic] = {'accuracy': k / n, 'ci': wilson_ci(k, n), 'n': n}
    return dict(sorted(result.items(), key=lambda x: x[1]['accuracy'], reverse=True))
