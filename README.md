# 02445 Group 11 - Evaluating Gemma on 02450 MCQ exams

Project for DTU 02445 (Statistical Evaluation of AI), June 2026.

We test the language model Gemma 4 on multiple choice questions from old DTU
02450 (Intro to Machine Learning) exams. The main question: does Gemma get
worse when it has to read a figure or table as an image, compared to getting
the same information as text?

Report deadline: 24.06.2026. Seminar: 25-26.06.2026.

## Setup

Gemma runs locally through the transformers library. collect.py picks the
device by itself: an NVIDIA GPU (cuda) if there is one, otherwise Apple
Silicon (mps), otherwise the CPU.

On a PC with an NVIDIA GPU, install the CUDA build of torch FIRST (otherwise
pip installs the CPU-only version on Windows):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

On a Mac, just:

```bash
pip install -r requirements.txt
```

Gemma is a gated model, so the machine needs a Hugging Face login once:
accept the licence on the google/gemma-4-E2B-it model page, then run

```bash
hf auth login
```

Quick check that the model loads and answers before the full run:

```bash
python simple_gemma.py
```

## How to run

```bash
python build_questions_csv.py   # data/encoded/*.json -> data/questions.csv
python collect.py               # Gemma answers every question -> data/results.csv
python summarize.py             # counts and accuracies from the answers
```

## The data

The 15 exams (27 questions each) are transcribed in data/encoded/, one JSON
file per exam. build_questions_csv.py turns them into data/questions.csv with
one row per (question, modality). The figure/table crops in data/screenshots/
are cut out of the exam PDFs by crop_figures.py.

Question types:

| Type | Meaning | Asked as |
|------|---------|----------|
| A | pure text, no figure needed | text |
| B | needs a figure/graph | screenshot (+ text_desc if the figure can be written as text) |
| C | needs a table/matrix | screenshot + text_desc |

Modalities:

- text: the question as plain text (type A)
- screenshot: the question as text plus the figure/table as a PNG image
- text_desc: the same figure/table written out as text instead of an image

Every question is sent to Gemma as a brand new call with no chat history, so
the answers are independent of each other. Gemma is prompted to reason step
by step (chain of thought) and to close its reply with a "FINAL ANSWER: X"
line, which collect.py parses from the END of the reply. Decoding is greedy,
so the answers stay reproducible. (Answers collected with the older
single-letter/no-reasoning prompt are kept in data/results_noCoT_archive.csv
and are not comparable with the current protocol.)

## Step 1: collect Gemma's answers

```bash
python collect.py                    # everything not done yet
python collect.py --exam Fall2024    # one exam only
python collect.py --modality text    # one modality only
python collect.py --dry-run          # show the prompts without running Gemma
```

Answers are appended to data/results.csv. Questions that are already answered
get skipped, so the script is safe to stop and rerun.

## Step 2: count the results

This repo stops at the counts: it translates the exams, feeds them to Gemma,
and tallies the answers. The statistical analysis of those numbers is done by
us separately and is described in the report.

```bash
python summarize.py                    # uses data/results.csv
python summarize.py data/example.csv   # try it on the example data
python make_figure.py                  # bar chart of the accuracies
```

summarize.py prints, in order: accuracy per modality; the pair counts for the
questions asked both as image and as text (both right / both wrong / only
image right / only text right); the pure-text and pure-graph groups; accuracy
per question type in text form; and how often Gemma answered E ("don't know")
on image vs text input. The outcome per question is simply correct/wrong.

## Files

```
collect.py               asks Gemma, saves answers to data/results.csv
helpers.py               small functions shared by summarize.py and make_figure.py
summarize.py             counts and accuracies from data/results.csv
make_figure.py           bar chart of the accuracies per group
crop_figures.py          cuts the figures/tables out of the exam PDFs
build_questions_csv.py   data/encoded/*.json -> data/questions.csv
validate_encoded.py      sanity check of the encoded JSONs
apply_corrections.py     fixes we made after double-checking the encodings
data/
  encoded/               the 15 exams as JSON (the master copy)
  questions.csv          what collect.py reads (generated, don't edit by hand)
  results.csv            Gemma's answers (made by collect.py)
  example.csv            example data: python summarize.py data/example.csv
  screenshots/           figure/table crops (made by crop_figures.py)
ML-examsets/             the exam PDFs
ML-solutions/            the solution PDFs
```
