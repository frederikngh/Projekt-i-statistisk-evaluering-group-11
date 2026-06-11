"""summarize.py - counts and accuracies from Gemma's answers.

This repo only translates the exams, collects Gemma's answers and counts
them. The statistical analysis of these numbers (tests, confidence
intervals, power) is done by us separately - this script prints every
number that analysis needs.

Usage:
    python summarize.py                    # uses data/results.csv
    python summarize.py data/example.csv   # try it on the example data
"""

from helpers import load_results, subset, count_correct, percent, acc_text
from helpers import text_A_rows, graph_only_rows, mcnemar_counts


def print_pairs(rows, types, label):
    """The four pair counts for questions asked both as image and as text."""
    both, only_image, only_text, neither = mcnemar_counts(rows, types)
    n = both + only_image + only_text + neither
    if n == 0:
        print("  " + label + ": no paired questions")
        return
    image_right = both + only_image
    text_right = both + only_text
    print("  " + label + ": " + str(n) + " pairs")
    print("     both right: " + str(both) + ", both wrong: " + str(neither))
    print("     only image right: " + str(only_image) + ", only text right: " + str(only_text))
    print("     correct as image: " + str(image_right) + "/" + str(n)
          + ", correct as text: " + str(text_right) + "/" + str(n))


rows = load_results()

print()
print("=== 1. ACCURACY PER MODALITY ===")
modalities = ["text", "text_desc", "screenshot"]
for modality in modalities:
    sub = subset(rows, "modality", modality)
    if len(sub) > 0:
        print("  " + modality + ": " + acc_text(sub))

print()
print("=== 2. PAIRED QUESTIONS (same question as image AND as text) ===")
print_pairs(rows, ["B"], "Type B (figures)")
print_pairs(rows, ["C"], "Type C (tables)")
print_pairs(rows, ["B", "C"], "Type B+C combined")

print()
print("=== 3. PURE TEXT vs PURE GRAPH (two separate question groups) ===")
text_qs = text_A_rows(rows)
graph_qs = graph_only_rows(rows)
if len(text_qs) > 0:
    print("  pure text (type A):        " + acc_text(text_qs))
if len(graph_qs) > 0:
    print("  pure graph (image-only B): " + acc_text(graph_qs))

print()
print("=== 4. QUESTION TYPES A/B/C, ALL IN TEXT FORM ===")
a_rows = text_A_rows(rows)
desc_rows = subset(rows, "modality", "text_desc")
b_rows = subset(desc_rows, "question_type", "B")
c_rows = subset(desc_rows, "question_type", "C")
if len(a_rows) > 0:
    print("  A (pure text): " + acc_text(a_rows))
if len(b_rows) > 0:
    print("  B (figures):   " + acc_text(b_rows))
if len(c_rows) > 0:
    print("  C (tables):    " + acc_text(c_rows))

print()
print("=== 5. E (\"DON'T KNOW\") ANSWERS, IMAGE vs TEXT INPUT ===")
img = subset(rows, "modality", "screenshot")
txt = []
for r in rows:
    if r["modality"] != "screenshot":
        txt.append(r)
e_img = len(subset(img, "gemma_answer", "E"))
e_txt = len(subset(txt, "gemma_answer", "E"))
if len(img) > 0:
    print("  image input: " + str(e_img) + "/" + str(len(img)) + " answers were E (" + percent(e_img / len(img)) + ")")
if len(txt) > 0:
    print("  text input:  " + str(e_txt) + "/" + str(len(txt)) + " answers were E (" + percent(e_txt / len(txt)) + ")")
