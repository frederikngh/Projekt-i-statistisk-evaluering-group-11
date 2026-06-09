# 02445 Group 11 — Evaluating Gemma on 02450 MCQ Exams

Statistical evaluation of the Gemma 4 language model on DTU 02450 (Intro to ML) MCQ exam questions.

**Deadline:** 24.06.2026 | **Seminar:** 25–26.06.2026

---

## Setup

```bash
pip install -r requirements.txt
```

Gemma runs locally via Transformers on Apple Silicon (MPS). Change `device="mps"` to `"cuda"` or `"cpu"` in `collect.py` if needed.

---

## Workflow

```
1. Fill in data/questions.csv   ← one row per (question, modality)
2. python collect.py            ← Gemma answers each question, saves to data/results.csv
3. python analyze.py            ← statistical report + figures
```

---

## Step 1 — Fill in data/questions.csv

Run `python collect.py` once to auto-generate the template, then fill it in.

| Column | Values | Notes |
|---|---|---|
| `exam_year` | `Fall2024`, `Dec2022`, … | Which exam set |
| `question_id` | `Q1` … `Q27` | Question number |
| `question_type` | `A` / `B` / `C` | See below |
| `topic` | `PCA`, `KNN`, … | Topic label |
| `modality` | `text` / `screenshot` / `text_desc` | See below |
| `question_text` | Full question text + options A–D | Always required |
| `description` | Text description of the figure/table | Required for `text_desc` modality |
| `image_path` | Path to screenshot PNG | Required for `screenshot` modality |
| `correct_answer` | `A` / `B` / `C` / `D` | Ground truth from solution PDF |

### Question types
| Type | Description | Modalities to test |
|---|---|---|
| **A** | Pure text — no figures needed | `text` only |
| **B** | Requires reading a figure/graph | `screenshot` AND `text_desc` |
| **C** | Requires reading a table/matrix | `screenshot` AND `text_desc` |

### Modalities
- `text` — question sent as plain text (Type A)
- `screenshot` — question sent as a PNG image to Gemma's vision input
- `text_desc` — the figure/table written out as text instead of an image

> **Independence:** Gemma is called fresh for every (question, modality) pair — no shared conversation history.

---

## Step 2 — Collect Gemma answers

```bash
python collect.py                        # run all pending questions
python collect.py --exam Fall2024        # only one exam set
python collect.py --dry-run              # preview prompts without querying Gemma
python collect.py --question Q1 --exam Fall2024 --modality screenshot
```

Results are appended to `data/results.csv`. Already-completed pairs are skipped, so you can stop and resume safely.

---

## Step 3 — Run statistical analysis

```bash
python analyze.py                        # uses data/results.csv
python analyze.py data/example.csv       # test with example data
python analyze.py --no-plots             # skip figure generation
```

Figures are saved to `figures/`.

---

## Statistical tests

| Test | Research question | Cheat-sheet path |
|---|---|---|
| **Binomial test** | Is Gemma significantly better than 25% chance? | Description of one group → proportion |
| **McNemar's test** | Does modality (screenshot vs. text desc) affect accuracy? | Compare two groups → Paired → Nominal → McNemar |
| **Chi-square / Fisher** | Does accuracy differ across question types A, B, C? | Compare 3+ groups → Unmatched → Proportions → Chi-square |
| **Wilson CIs** | Uncertainty on each accuracy estimate | — |

### McNemar's test — the key paired test

For each Type B/C question answered in both modalities:

```
                    text_desc ✓   text_desc ✗
screenshot ✓           a              b       ← screenshot wins (discordant)
screenshot ✗           c              d       ← text_desc wins (discordant)
```

H₀: b = c (both modalities equally accurate). Uses exact binomial when b+c < 25.

---

## File structure

```
collect.py          ← Query Gemma, save to results.csv
analyze.py          ← Run statistical analysis, generate figures
requirements.txt
data/
  questions.csv     ← Fill this in (auto-generated on first run of collect.py)
  results.csv       ← Auto-filled by collect.py
  example.csv       ← Example data — try: python analyze.py data/example.csv
  screenshots/      ← Put question screenshot PNGs here
src/
  loader.py         ← CSV loading + validation
  tests.py          ← Statistical tests (binomial, McNemar, chi-square)
  plots.py          ← All visualisations
figures/            ← Auto-created by analyze.py
ML-examsets/        ← Exam PDFs
ML-solutions/       ← Solution PDFs
```
