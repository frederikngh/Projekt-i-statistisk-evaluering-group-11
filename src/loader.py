import pandas as pd
from pathlib import Path

REQUIRED_COLUMNS = [
    'exam_year', 'question_id', 'question_type',
    'topic', 'modality', 'gemma_answer', 'correct_answer',
]
VALID_TYPES = {'A', 'B', 'C'}
VALID_MODALITIES = {'text', 'screenshot', 'text_desc'}
VALID_ANSWERS = {'A', 'B', 'C', 'D', 'E'}


def load_results(path: str | Path) -> pd.DataFrame:
    """
    Load the results CSV and derive the is_correct column.

    CSV format — one row per (question, modality):
      exam_year    : e.g. "Fall2024", "Dec2022"
      question_id  : e.g. "Q1", "Q17"
      question_type: A (pure text), B (figure), C (table/matrix)
      topic        : e.g. "PCA", "KNN", "Clustering"
      modality     : text | screenshot | text_desc
      gemma_answer : A | B | C | D | E
      correct_answer: A | B | C | D
    """
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df['is_correct'] = (df['gemma_answer'] == df['correct_answer']).astype(int)
    return df


def validate_data(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    bad_types = set(df['question_type'].unique()) - VALID_TYPES
    if bad_types:
        raise ValueError(f"Unknown question_type values: {bad_types}. Use A, B, or C.")

    bad_mod = set(df['modality'].unique()) - VALID_MODALITIES
    if bad_mod:
        raise ValueError(f"Unknown modality values: {bad_mod}. Use: text, screenshot, text_desc.")

    bad_ans = set(df['gemma_answer'].unique()) - VALID_ANSWERS
    if bad_ans:
        raise ValueError(f"Unknown gemma_answer values: {bad_ans}. Use A–E.")

    # Warn if Type A questions have non-text modality
    a_bad = df[(df['question_type'] == 'A') & (df['modality'] != 'text')]
    if len(a_bad):
        print(f"  Warning: {len(a_bad)} Type-A rows have modality != 'text'.")

    # Warn if Type B/C questions have 'text' modality (should be screenshot or text_desc)
    bc_bad = df[(df['question_type'].isin(['B', 'C'])) & (df['modality'] == 'text')]
    if len(bc_bad):
        print(f"  Warning: {len(bc_bad)} Type B/C rows have modality='text'.")

    print(f"  Data OK: {len(df)} rows, "
          f"{df.groupby(['exam_year', 'question_id']).ngroups} unique questions.")


def primary_observations(df: pd.DataFrame) -> pd.DataFrame:
    """
    One canonical observation per question for between-type comparisons.
    Type A → modality='text'
    Type B/C → modality='text_desc'  (same text-based input, comparable across types)
    Falls back to all rows if the expected modalities are missing.
    """
    mask = (
        ((df['question_type'] == 'A') & (df['modality'] == 'text')) |
        ((df['question_type'].isin(['B', 'C'])) & (df['modality'] == 'text_desc'))
    )
    result = df[mask]
    return result if len(result) > 0 else df
