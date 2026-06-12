"""helpers.py - small functions shared by summarize.py and make_figure.py.

The results file can be given on the command line, e.g.
    python summarize.py data/example.csv
otherwise data/results.csv is used.
"""

import csv
import os
import sys

CHANCE = 0.25   # 4 real options (A-D) -> random guessing gives 25%

# The newest exam (December 2025) is kept OUT of the main analyses and is only
# used for the training-data contamination check: it is the one exam we know
# Gemma cannot have seen during training.
CLEAN_EXAM = "Fall2025"


def load_results():
    """Read the results CSV and add r["correct"] (True/False) to every row."""
    path = "data/results.csv"
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if not os.path.exists(path):
            print("ERROR:", path, "not found.")
            sys.exit(1)
    if not os.path.exists(path):
        print(path, "not found - using the example data instead.")
        path = "data/example.csv"
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    for r in rows:
        if r["gemma_answer"] == r["correct_answer"]:
            r["correct"] = True
        else:
            r["correct"] = False
    return rows


def main_rows(rows):
    """The rows from the 15 original exams (the main dataset)."""
    out = []
    for r in rows:
        if r["exam_year"] != CLEAN_EXAM:
            out.append(r)
    return out


def clean_rows(rows):
    """The rows from the held-out clean exam (Fall 2025)."""
    out = []
    for r in rows:
        if r["exam_year"] == CLEAN_EXAM:
            out.append(r)
    return out


def subset(rows, column, value):
    """All rows where row[column] == value."""
    out = []
    for r in rows:
        if r[column] == value:
            out.append(r)
    return out


def count_correct(rows):
    """How many rows Gemma answered correctly."""
    k = 0
    for r in rows:
        if r["correct"]:
            k = k + 1
    return k


def percent(x):
    """A number between 0 and 1 written as a percent, e.g. 0.818 -> '81.8%'."""
    return str(round(100 * x, 1)) + "%"


def acc_text(rows):
    """Accuracy written out, e.g. '81.8% (9/11)'."""
    k = count_correct(rows)
    n = len(rows)
    return percent(k / n) + " (" + str(k) + "/" + str(n) + ")"


def text_A_rows(rows):
    """The pure-text answers: type-A questions asked in the text modality."""
    out = []
    for r in rows:
        if r["modality"] == "text" and r["question_type"] == "A":
            out.append(r)
    return out


def graph_only_rows(rows):
    """Screenshot answers to figures with NO text version (scatter,
    dendrogram, ...) - pure graph reading."""
    # first collect the ids of all questions that DO have a text version
    has_text = []
    for r in rows:
        if r["modality"] == "text_desc":
            has_text.append(r["exam_year"] + " " + r["question_id"])
    # then keep the type-B screenshot answers that are not in that list
    graph = []
    for r in rows:
        if r["question_type"] == "B" and r["modality"] == "screenshot":
            key = r["exam_year"] + " " + r["question_id"]
            if key not in has_text:
                graph.append(r)
    return graph


def mcnemar_counts(rows, types):
    """For questions (of the given types) answered BOTH as image and as text:
    count (both right, only image right, only text right, both wrong)."""
    image_ok = {}
    text_ok = {}
    for r in rows:
        if r["question_type"] in types:
            key = r["exam_year"] + " " + r["question_id"]
            if r["modality"] == "screenshot":
                image_ok[key] = r["correct"]
            if r["modality"] == "text_desc":
                text_ok[key] = r["correct"]
    both = 0
    only_image = 0
    only_text = 0
    neither = 0
    for key in image_ok:
        if key in text_ok:
            if image_ok[key] and text_ok[key]:
                both = both + 1
            elif image_ok[key]:
                only_image = only_image + 1
            elif text_ok[key]:
                only_text = only_text + 1
            else:
                neither = neither + 1
    return both, only_image, only_text, neither
