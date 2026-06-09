"""Test 3: pure-text questions vs. pure-graph questions.  (two-proportion z-test)

Type-A text questions vs. the geometric figures (scatter, dendrogram, ...)
that exist only as an image. H0: same accuracy in both groups.
UNPAIRED: the two groups contain different questions, so question
difficulty is not controlled - we say so in the report.
(On a 2x2 table the z-test equals the chi-square test: z^2 = chi2.)
"""

from statsmodels.stats.proportion import proportions_ztest

from helpers import load_results, text_A_rows, graph_only_rows
from helpers import count_correct, acc_text, fmt_p

rows = load_results()
text_qs = text_A_rows(rows)
graph_qs = graph_only_rows(rows)

print()
print("--- 3. PURE-TEXT vs. PURE-GRAPH QUESTIONS (z-test, unpaired) ---")
if len(text_qs) == 0 or len(graph_qs) == 0:
    print("  not enough data (need type-A text rows and image-only figure rows)")
else:
    correct_counts = [count_correct(text_qs), count_correct(graph_qs)]
    group_sizes = [len(text_qs), len(graph_qs)]
    z, p = proportions_ztest(correct_counts, group_sizes)
    print("  text : " + acc_text(text_qs))
    print("  graph: " + acc_text(graph_qs))
    print("  z = " + str(round(z, 3)) + "   " + fmt_p(p))
