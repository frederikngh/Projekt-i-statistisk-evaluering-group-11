"""
Visualizations for the Gemma MCQ evaluation project.
All functions accept a DataFrame and a save path (or None to display).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from .loader import primary_observations
from .tests import topic_accuracy, mcnemar_modality_test, type_comparison_test, wilson_ci

# ── Shared style ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

TYPE_COLORS = {'A': '#4C72B0', 'B': '#DD8452', 'C': '#55A868'}
MOD_COLORS  = {'screenshot': '#C44E52', 'text_desc': '#4C72B0', 'text': '#4C72B0'}
CHANCE_COLOR = '#999999'


# ──────────────────────────────────────────────────────────────
#  1. Accuracy overview: overall + by question type
# ──────────────────────────────────────────────────────────────

def plot_accuracy_overview(df: pd.DataFrame, save_path: Path | None = None):
    """
    Bar chart showing overall accuracy and accuracy per question type,
    with 95% Wilson confidence interval error bars and a chance-level line.
    """
    primary = primary_observations(df)

    categories = {}
    # Overall
    k, n = int(primary['is_correct'].sum()), len(primary)
    ci = wilson_ci(k, n)
    categories['Overall'] = {'acc': k / n, 'ci': ci, 'color': '#2d6a9f', 'n': n}

    # Per type
    for qtype in sorted(primary['question_type'].unique()):
        sub = primary[primary['question_type'] == qtype]
        k, n = int(sub['is_correct'].sum()), len(sub)
        ci = wilson_ci(k, n)
        label = {'A': 'Type A\n(Text)', 'B': 'Type B\n(Figure)', 'C': 'Type C\n(Table/Matrix)'}[qtype]
        categories[label] = {'acc': k / n, 'ci': ci, 'color': TYPE_COLORS[qtype], 'n': n}

    labels = list(categories.keys())
    accs   = [v['acc'] for v in categories.values()]
    errs_lo = [v['acc'] - v['ci'][0] for v in categories.values()]
    errs_hi = [v['ci'][1] - v['acc'] for v in categories.values()]
    colors  = [v['color'] for v in categories.values()]
    ns      = [v['n'] for v in categories.values()]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, accs, color=colors, alpha=0.85, width=0.5, zorder=3)
    ax.errorbar(
        range(len(labels)), accs,
        yerr=[errs_lo, errs_hi],
        fmt='none', color='black', capsize=5, linewidth=1.5, zorder=4,
    )

    # Chance line
    ax.axhline(0.25, color=CHANCE_COLOR, linestyle='--', linewidth=1.4, label='Chance (25%)', zorder=2)

    # Annotate bars with accuracy + n
    for i, (bar, acc, n) in enumerate(zip(bars, accs, ns)):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + errs_hi[i] + 0.015,
                f'{acc:.0%}\n(n={n})', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Accuracy (proportion correct)', fontsize=11)
    ax.set_title('Gemma Accuracy on 02450 MCQ Exams', fontsize=13, fontweight='bold')
    ax.set_ylim(0, min(1.15, max(accs) + 0.25))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.35, zorder=0)

    plt.tight_layout()
    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────────────────────
#  2. McNemar contingency heatmaps (one per type B, C, B+C)
# ──────────────────────────────────────────────────────────────

def plot_mcnemar_heatmaps(df: pd.DataFrame, save_dir: Path | None = None):
    """
    2×2 heatmap of the McNemar contingency table for each modality comparison.
    The discordant cells (b, c) drive the test.
    """
    mcnemar_results = mcnemar_modality_test(df)
    if not mcnemar_results:
        print("  No McNemar results to plot (need paired data).")
        return

    for label, result in mcnemar_results.items():
        table = result.details['table']
        ann = np.array([
            [f'a = {table[0,0]}\n(both correct)',   f'b = {table[0,1]}\n(screenshot ✓, text ✗)'],
            [f'c = {table[1,0]}\n(screenshot ✗, text ✓)', f'd = {table[1,1]}\n(both wrong)'],
        ])

        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(
            table.astype(float), annot=ann, fmt='', ax=ax,
            cmap='Blues', linewidths=1.5, linecolor='white',
            annot_kws={'size': 10}, cbar=False,
            xticklabels=['Text desc. ✓', 'Text desc. ✗'],
            yticklabels=['Screenshot ✓', 'Screenshot ✗'],
        )

        ax.set_title(
            f"McNemar Contingency Table — Type {label}\n"
            f"p = {result.p_value:.4f}  |  b+c = {result.details['discordant']}",
            fontsize=11, fontweight='bold',
        )
        ax.set_xlabel('Text Description Modality', fontsize=10)
        ax.set_ylabel('Screenshot Modality', fontsize=10)

        plt.tight_layout()
        path = (save_dir / f'mcnemar_type_{label.replace("+", "")}.png') if save_dir else None
        _save_or_show(fig, path)


# ──────────────────────────────────────────────────────────────
#  3. Accuracy by topic (horizontal bar chart)
# ──────────────────────────────────────────────────────────────

def plot_accuracy_by_topic(df: pd.DataFrame, save_path: Path | None = None):
    """Horizontal bar chart of accuracy per topic, sorted descending."""
    topics = topic_accuracy(df)
    if not topics:
        return

    labels = list(topics.keys())
    accs   = [v['accuracy'] for v in topics.values()]
    errs_lo = [v['accuracy'] - v['ci'][0] for v in topics.values()]
    errs_hi = [v['ci'][1] - v['accuracy'] for v in topics.values()]
    ns      = [v['n'] for v in topics.values()]

    fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.45 + 1.5)))
    y_pos = range(len(labels))
    bars = ax.barh(y_pos, accs, xerr=[errs_lo, errs_hi],
                   color='#4C72B0', alpha=0.82, height=0.6,
                   error_kw={'capsize': 4, 'linewidth': 1.4})
    ax.axvline(0.25, color=CHANCE_COLOR, linestyle='--', linewidth=1.4, label='Chance (25%)')

    for i, (acc, n) in enumerate(zip(accs, ns)):
        ax.text(acc + errs_hi[i] + 0.01, i, f'{acc:.0%} (n={n})',
                va='center', fontsize=8.5)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Accuracy', fontsize=10)
    ax.set_title('Accuracy per Topic (primary modality)', fontsize=12, fontweight='bold')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.set_xlim(0, min(1.4, max(accs) + 0.3))
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.35)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────────────────────
#  4. Modality comparison: grouped bars for B and C types
# ──────────────────────────────────────────────────────────────

def plot_modality_comparison(df: pd.DataFrame, save_path: Path | None = None):
    """
    Grouped bar chart: screenshot vs. text_desc accuracy for Type B and Type C.
    Shows how modality affects performance within each question type.
    """
    bc = df[df['question_type'].isin(['B', 'C']) &
            df['modality'].isin(['screenshot', 'text_desc'])]

    if bc.empty:
        return

    groups = bc.groupby(['question_type', 'modality'])['is_correct'].agg(['sum', 'count'])
    groups.columns = ['correct', 'n']
    groups['acc'] = groups['correct'] / groups['n']
    groups['ci_lo'] = groups.apply(lambda r: wilson_ci(int(r['correct']), int(r['n']))[0], axis=1)
    groups['ci_hi'] = groups.apply(lambda r: wilson_ci(int(r['correct']), int(r['n']))[1], axis=1)
    groups = groups.reset_index()

    qtypes = sorted(groups['question_type'].unique())
    modalities = ['screenshot', 'text_desc']
    x = np.arange(len(qtypes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, mod in enumerate(modalities):
        sub = groups[groups['modality'] == mod].set_index('question_type')
        accs   = [sub.loc[t, 'acc'] if t in sub.index else 0 for t in qtypes]
        errs_lo = [sub.loc[t, 'acc'] - sub.loc[t, 'ci_lo'] if t in sub.index else 0 for t in qtypes]
        errs_hi = [sub.loc[t, 'ci_hi'] - sub.loc[t, 'acc'] if t in sub.index else 0 for t in qtypes]
        label = 'Screenshot' if mod == 'screenshot' else 'Text Description'
        ax.bar(x + i * width, accs, width, label=label,
               color=MOD_COLORS[mod], alpha=0.85, zorder=3)
        ax.errorbar(x + i * width, accs, yerr=[errs_lo, errs_hi],
                    fmt='none', color='black', capsize=4, linewidth=1.3, zorder=4)

    ax.axhline(0.25, color=CHANCE_COLOR, linestyle='--', linewidth=1.4, label='Chance (25%)')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([f'Type {t}' for t in qtypes], fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.set_title('Screenshot vs. Text Description Accuracy\n(Type B & C questions)', fontsize=12, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.35, zorder=0)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────────────────────
#  5. Per-question answer heatmap
# ──────────────────────────────────────────────────────────────

def plot_answer_heatmap(df: pd.DataFrame, exam_year: str, save_path: Path | None = None):
    """
    Heatmap for a single exam: rows = questions, columns = answer options A–E.
    Cell colour: green = Gemma's correct answer, red = Gemma's wrong answer,
    grey = not chosen.
    """
    exam = df[df['exam_year'] == exam_year].copy()
    if exam.empty:
        print(f"  No data for exam year: {exam_year}")
        return

    # Use one row per question (primary modality)
    from .loader import primary_observations
    exam = primary_observations(exam)

    questions = sorted(exam['question_id'].unique(),
                       key=lambda q: int(''.join(filter(str.isdigit, q)) or 0))
    options = ['A', 'B', 'C', 'D', 'E']

    # Build matrix: 0=not chosen, 1=correct, -1=wrong
    matrix = pd.DataFrame(0, index=questions, columns=options)
    for _, row in exam.iterrows():
        q, ans, correct = row['question_id'], row['gemma_answer'], row['correct_answer']
        if q in matrix.index and ans in options:
            matrix.loc[q, ans] = 1 if row['is_correct'] else -1

    fig, ax = plt.subplots(figsize=(8, max(5, len(questions) * 0.38 + 1.5)))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(matrix.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(len(options)))
    ax.set_xticklabels(options, fontsize=10, fontweight='bold')
    ax.set_yticks(range(len(questions)))
    ax.set_yticklabels(questions, fontsize=8)
    ax.set_xlabel("Gemma's Answer", fontsize=10)
    ax.set_title(f'Gemma Answers — {exam_year}\n(green=correct, red=wrong, white=not chosen)',
                 fontsize=11, fontweight='bold')

    # Annotate non-zero cells
    for i, q in enumerate(questions):
        for j, opt in enumerate(options):
            val = matrix.loc[q, opt]
            if val != 0:
                symbol = '✓' if val == 1 else '✗'
                ax.text(j, i, symbol, ha='center', va='center', fontsize=10,
                        color='black', fontweight='bold')

    plt.colorbar(im, ax=ax, shrink=0.5, label='Correct (1) / Wrong (-1)')
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ──────────────────────────────────────────────────────────────
#  Master save function
# ──────────────────────────────────────────────────────────────

def save_all_plots(df: pd.DataFrame, figures_dir: Path):
    figures_dir.mkdir(exist_ok=True)

    plot_accuracy_overview(df, figures_dir / 'accuracy_overview.png')
    print(f"    Saved: accuracy_overview.png")

    plot_modality_comparison(df, figures_dir / 'modality_comparison.png')
    print(f"    Saved: modality_comparison.png")

    plot_accuracy_by_topic(df, figures_dir / 'accuracy_by_topic.png')
    print(f"    Saved: accuracy_by_topic.png")

    plot_mcnemar_heatmaps(df, figures_dir)
    print(f"    Saved: mcnemar_type_*.png")

    for year in df['exam_year'].unique():
        fname = f'answers_{year.replace(" ", "_")}.png'
        plot_answer_heatmap(df, year, figures_dir / fname)
        print(f"    Saved: {fname}")


# ──────────────────────────────────────────────────────────────
#  Utility
# ──────────────────────────────────────────────────────────────

def _save_or_show(fig: plt.Figure, path: Path | None):
    if path:
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
