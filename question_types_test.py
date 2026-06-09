"""Test 4: does accuracy differ across question types A/B/C?  (chi-square)

Everything is in TEXT form (A as plain text, B/C as the text description),
so only the question type changes, not the input modality.
H0: all three types have the same accuracy.
Caveat: only the few 'describable' type-B figures have a text version,
so type B is a small, non-random subset here.
"""

from scipy.stats import chi2_contingency

from helpers import load_results, subset, text_A_rows, count_correct, acc_text, fmt_p

rows = load_results()

a_rows = text_A_rows(rows)
desc_rows = subset(rows, "modality", "text_desc")
b_rows = subset(desc_rows, "question_type", "B")
c_rows = subset(desc_rows, "question_type", "C")

print()
print("--- 4. ACCURACY ACROSS QUESTION TYPES A/B/C (chi-square) ---")
table = []
if len(a_rows) > 0:
    print("  A (pure text): " + acc_text(a_rows))
    table.append([count_correct(a_rows), len(a_rows) - count_correct(a_rows)])
if len(b_rows) > 0:
    print("  B (figures):   " + acc_text(b_rows))
    table.append([count_correct(b_rows), len(b_rows) - count_correct(b_rows)])
if len(c_rows) > 0:
    print("  C (tables):    " + acc_text(c_rows))
    table.append([count_correct(c_rows), len(c_rows) - count_correct(c_rows)])

if len(table) >= 2:
    chi2, p, dof, expected = chi2_contingency(table)
    print("  chi2 = " + str(round(chi2, 3)) + ", df = " + str(dof) + "   " + fmt_p(p))
    if expected.min() < 5:
        print("  NOTE: smallest expected cell is " + str(round(expected.min(), 1)) + " (< 5),")
        print("  so the chi-square assumption is violated - interpret with care")
else:
    print("  not enough groups for a test")
