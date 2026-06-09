"""
collect.py — Automated Gemma querying for the 02450 MCQ evaluation.

Reads a question manifest (data/questions.csv), queries Gemma for each
(question, modality) pair, and appends results to data/results.csv.

Usage:
    python collect.py                        # run all pending questions
    python collect.py --exam Fall2024        # only one exam set
    python collect.py --dry-run              # print prompts without querying Gemma
    python collect.py --question Q1 --exam Fall2024 --modality screenshot

Gemma is loaded once and reused. Each question gets a FRESH pipeline call
with no conversation history — this is required for statistical independence.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────
#  Prompt templates
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are answering a multiple-choice exam question from a university "
    "Machine Learning course. Each question has exactly one correct answer "
    "among A, B, C, D. You may also answer E if you genuinely do not know. "
    "Respond with ONLY a single capital letter: A, B, C, D, or E. "
    "Do not explain your answer."
)

TEXT_TEMPLATE = """{system}

Question:
{question_text}

Your answer (A/B/C/D/E):"""

IMAGE_TEMPLATE = """{system}

Question:
{question_text}

The figure(s)/table(s) this question refers to are in the attached image.

Your answer (A/B/C/D/E):"""

TEXT_DESC_TEMPLATE = """{system}

The question includes a figure/table that has been described in text below.
Use this description to answer the question.

{question_text}

Figure/Table Description:
{description}

Your answer (A/B/C/D/E):"""


# ──────────────────────────────────────────────────────────────
#  Gemma interface
# ──────────────────────────────────────────────────────────────

def load_gemma():
    """Load Gemma once. Requires: pip install transformers torch"""
    from transformers import pipeline
    print("Loading Gemma (this takes ~30s the first time)...", flush=True)
    model = pipeline(
        "image-text-to-text",
        model="google/gemma-4-E4B-it",
        dtype="auto",
        device="mps",       # Mac Apple Silicon — change to "cuda" or "cpu" if needed
    )
    print("Gemma loaded.", flush=True)
    return model


def query_gemma(model, prompt_text: str, image_path: str | None = None) -> str:
    """
    Send one isolated query to Gemma and extract the answer letter.
    Returns 'A', 'B', 'C', 'D', or 'E'.
    No conversation history is passed — each call is fully independent.
    """
    content = []
    if image_path:
        content.append({"type": "image", "url": str(Path(image_path).resolve())})
    content.append({"type": "text", "text": prompt_text})

    messages = [{"role": "user", "content": content}]

    out = model(text=messages, max_new_tokens=16)
    raw = out[0]["generated_text"][-1]["content"].strip()
    return _extract_letter(raw)


def _extract_letter(text: str) -> str:
    """Extract the first A/B/C/D/E from Gemma's response."""
    match = re.search(r'\b([ABCDE])\b', text.upper())
    if match:
        return match.group(1)
    # Fallback: first character if it's a letter
    first = text.strip()[:1].upper()
    return first if first in 'ABCDE' else 'E'


# ──────────────────────────────────────────────────────────────
#  Question manifest (data/questions.csv)
# ──────────────────────────────────────────────────────────────

QUESTIONS_SCHEMA = [
    'exam_year', 'question_id', 'question_type', 'topic',
    'modality', 'question_text', 'description', 'image_path', 'correct_answer',
]

RESULTS_SCHEMA = [
    'exam_year', 'question_id', 'question_type', 'topic',
    'modality', 'gemma_answer', 'correct_answer',
]

QUESTIONS_PATH = Path("data/questions.csv")
RESULTS_PATH   = Path("data/results.csv")


def load_questions(exam_filter: str | None = None) -> list[dict]:
    if not QUESTIONS_PATH.exists():
        _create_questions_template()
        print(f"\nCreated template: {QUESTIONS_PATH}")
        print("Fill it in with your questions, then re-run collect.py.")
        sys.exit(0)

    rows = []
    with open(QUESTIONS_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if exam_filter and row.get('exam_year') != exam_filter:
                continue
            rows.append(row)
    return rows


def load_done_pairs() -> set[tuple]:
    """Return set of (exam_year, question_id, modality) already in results.csv."""
    done = set()
    if not RESULTS_PATH.exists():
        return done
    with open(RESULTS_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((row['exam_year'], row['question_id'], row['modality']))
    return done


def append_result(row: dict):
    write_header = not RESULTS_PATH.exists()
    with open(RESULTS_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_SCHEMA)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row[k] for k in RESULTS_SCHEMA})


# ──────────────────────────────────────────────────────────────
#  Template for questions.csv
# ──────────────────────────────────────────────────────────────

def _create_questions_template():
    QUESTIONS_PATH.parent.mkdir(exist_ok=True)
    example_rows = [
        # Type A: text only
        {
            'exam_year': 'Fall2024',
            'question_id': 'Q5',
            'question_type': 'A',
            'topic': 'LogisticRegression',
            'modality': 'text',
            'question_text': (
                'Which one of the following statements about logistic regression is correct?\n'
                'A. The attributes of a logistic regression model cannot be modified...\n'
                'B. In terms of bias-variance trade-off, a well-tuned model has negligible bias...\n'
                'C. For a fixed value of lambda, there exists a closed-form solution...\n'
                'D. If the estimated test error is approx 0.5, this indicates over-regularization.'
            ),
            'description': '',
            'image_path': '',
            'correct_answer': 'D',
        },
        # Type B: screenshot modality
        {
            'exam_year': 'Fall2024',
            'question_id': 'Q1',
            'question_type': 'B',
            'topic': 'PCA',
            'modality': 'screenshot',
            'question_text': 'Which one of the following matrices represents the empirical correlation matrix for these attributes? A. ... B. ... C. ... D. ...',
            'description': '',
            'image_path': 'data/screenshots/Fall2024_Q1.png',
            'correct_answer': 'C',
        },
        # Type B: text_desc modality (same question, text description of figure)
        {
            'exam_year': 'Fall2024',
            'question_id': 'Q1',
            'question_type': 'B',
            'topic': 'PCA',
            'modality': 'text_desc',
            'question_text': 'Which one of the following matrices represents the empirical correlation matrix for these attributes? A. ... B. ... C. ... D. ...',
            'description': (
                'The scatter plot matrix shows 4 attributes: x1 (Magnitude), x2 (Depth), '
                'x3 (Latitude), x4 (Longitude). x1 and x3 appear strongly correlated '
                '(tight diagonal pattern). x2 and x4 show moderate correlation. '
                'x1-x2 show negative correlation.'
            ),
            'image_path': '',
            'correct_answer': 'C',
        },
    ]

    with open(QUESTIONS_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=QUESTIONS_SCHEMA)
        writer.writeheader()
        writer.writerows(example_rows)


# ──────────────────────────────────────────────────────────────
#  Build prompt from question row
# ──────────────────────────────────────────────────────────────

def build_prompt(row: dict) -> tuple[str, str | None]:
    """Returns (prompt_text, image_path_or_None)."""
    modality = row['modality']
    qtext    = row['question_text'].strip()
    desc     = row.get('description', '').strip()
    img      = row.get('image_path', '').strip() or None

    if modality == 'text':
        prompt = TEXT_TEMPLATE.format(system=SYSTEM_PROMPT, question_text=qtext)
        return prompt, None

    elif modality == 'screenshot':
        if not img:
            raise ValueError(
                f"  {row['exam_year']} {row['question_id']}: "
                f"modality=screenshot but image_path is empty."
            )
        prompt = IMAGE_TEMPLATE.format(system=SYSTEM_PROMPT, question_text=qtext)
        return prompt, img

    elif modality == 'text_desc':
        if not desc:
            raise ValueError(
                f"  {row['exam_year']} {row['question_id']}: "
                f"modality=text_desc but description is empty."
            )
        prompt = TEXT_DESC_TEMPLATE.format(
            system=SYSTEM_PROMPT,
            question_text=qtext,
            description=desc,
        )
        return prompt, None

    else:
        raise ValueError(f"Unknown modality: {modality}")


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exam',     help='Filter to one exam year, e.g. Fall2024')
    parser.add_argument('--question', help='Filter to one question, e.g. Q1')
    parser.add_argument('--modality', help='Filter to one modality')
    parser.add_argument('--dry-run',  action='store_true', help='Print prompts, do not query Gemma')
    args = parser.parse_args()

    questions = load_questions(exam_filter=args.exam)

    if args.question:
        questions = [q for q in questions if q['question_id'] == args.question]
    if args.modality:
        questions = [q for q in questions if q['modality'] == args.modality]

    done = load_done_pairs()
    pending = [
        q for q in questions
        if (q['exam_year'], q['question_id'], q['modality']) not in done
    ]

    print(f"Questions in manifest : {len(questions)}")
    print(f"Already done          : {len(done)}")
    print(f"Pending               : {len(pending)}")

    if not pending:
        print("Nothing to do. Run analyze.py to see results.")
        return

    model = None
    if not args.dry_run:
        model = load_gemma()

    for i, row in enumerate(pending, 1):
        key = f"{row['exam_year']} {row['question_id']} [{row['modality']}]"
        print(f"\n[{i}/{len(pending)}] {key}", flush=True)

        try:
            prompt, img = build_prompt(row)
        except ValueError as e:
            print(f"  SKIP: {e}")
            continue

        if args.dry_run:
            print("  --- PROMPT ---")
            print(prompt[:300] + ('...' if len(prompt) > 300 else ''))
            if img:
                print(f"  Image: {img}")
            continue

        answer = query_gemma(model, prompt, image_path=img)
        correct = row['correct_answer'].strip().upper()
        status = '✓' if answer == correct else '✗'
        print(f"  Gemma: {answer}  Correct: {correct}  {status}", flush=True)

        append_result({**row, 'gemma_answer': answer})

    if not args.dry_run:
        print(f"\nDone. Results saved to {RESULTS_PATH}")
        print("Run: python analyze.py")


if __name__ == '__main__':
    main()
