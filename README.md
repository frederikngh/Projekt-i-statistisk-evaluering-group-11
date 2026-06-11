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
python run_all.py               # all statistical tests + the report figure
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

## Step 2: run the tests

Every test is its own small script, with shared functions in helpers.py:

```bash
python run_all.py                    # everything
python run_all.py data/example.csv   # try it out on the example data
python mcnemar_test.py               # or run a single test
```

| Script | Test | Question it answers |
|--------|------|---------------------|
| binomial_test.py | binomial test | is Gemma better than guessing (25%)? |
| mcnemar_test.py | McNemar's test (paired) | same question, image vs text: does modality matter? (our main test) |
| text_vs_graph_test.py | two-proportion z-test | pure text questions vs pure graph questions |
| question_types_test.py | chi-square | does accuracy differ between types A/B/C? |
| dont_know_test.py | chi-square 2x2 | does Gemma answer E ("don't know") more on images? |
| power_check.py | power | how big must an effect be before our tests can detect it? |

The outcome is simply correct/wrong, and every accuracy gets a 95% Wilson
confidence interval.

McNemar is the main test: every type B/C question with a faithful text version
is asked twice (figure as image, figure as text), so only the modality changes.
The test only uses the pairs where the two modalities disagree:

```
                 text right   text wrong
image right          a            b
image wrong          c            d
```

H0 is b = c. We use the exact (binomial) version of the test.

## Files

```
collect.py               asks Gemma, saves answers to data/results.csv
helpers.py               small functions shared by the test scripts
binomial_test.py         test 1: better than guessing?
mcnemar_test.py          test 2: image vs text, paired (the main test)
text_vs_graph_test.py    test 3: pure text vs pure graph
question_types_test.py   test 4: question types A/B/C
dont_know_test.py        test 5: how often Gemma answers E
power_check.py           how big an effect can we detect?
make_figure.py           the report figure (accuracy per group)
run_all.py               runs all of the above
crop_figures.py          cuts the figures/tables out of the exam PDFs
build_questions_csv.py   data/encoded/*.json -> data/questions.csv
validate_encoded.py      sanity check of the encoded JSONs
apply_corrections.py     fixes we made after double-checking the encodings
data/
  encoded/               the 15 exams as JSON (the master copy)
  questions.csv          what collect.py reads (generated, don't edit by hand)
  results.csv            Gemma's answers (made by collect.py)
  example.csv            example data: python run_all.py data/example.csv
  screenshots/           figure/table crops (made by crop_figures.py)
ML-examsets/             the exam PDFs
ML-solutions/            the solution PDFs
```
