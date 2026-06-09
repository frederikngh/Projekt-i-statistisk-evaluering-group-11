"""Power check: how big must an effect be before our tests can detect it?

This is about the DESIGN, not the results. It answers step 4d of the
project description ("how are you deciding on the number of prompts?").
"""

from scipy.stats import binomtest
from statsmodels.stats.proportion import power_proportions_2indep

from helpers import load_results, mcnemar_counts, text_A_rows, graph_only_rows
from helpers import count_correct

rows = load_results()

print()
print("--- 6. HOW BIG MUST AN EFFECT BE FOR US TO SEE IT? (power) ---")

# Part 1: McNemar. The test only uses the pairs where image and text DISAGREE.
# Under H0 every disagreement is a 50/50 coin flip, so we ask: out of n_d
# disagreements, how many must favour the same modality before p < 0.05?
both, only_image, only_text, neither = mcnemar_counts(rows, ["B", "C"])
n_pairs = both + only_image + only_text + neither
n_d = only_image + only_text
if n_d >= 5:
    needed = 0
    for k in range(n_d // 2 + 1, n_d + 1):   # from a bare majority up to all of them
        result = binomtest(k, n_d, 0.5)
        if result.pvalue < 0.05:
            needed = k
            break
    if needed > 0:
        print("  McNemar: " + str(n_pairs) + " pairs, of which " + str(n_d) + " disagree.")
        print("  At least " + str(needed) + " of those " + str(n_d) + " must favour the SAME modality for p < 0.05.")
    else:
        print("  McNemar: only " + str(n_d) + " disagreements - significance is impossible.")

# Part 2: z-test. Try a drop of 1%, 2%, 3%, ... below the text accuracy and ask
# the power function how likely the z-test is to detect that drop with our
# group sizes. We report the first drop where the power reaches 80%.
text_qs = text_A_rows(rows)
graph_qs = graph_only_rows(rows)
if len(text_qs) > 0 and len(graph_qs) > 0:
    p1 = count_correct(text_qs) / len(text_qs)
    for g in range(1, 80):
        gap = g / 100
        if p1 - gap <= 0:
            break
        result = power_proportions_2indep(gap, p1 - gap, len(text_qs),
                                          ratio=len(graph_qs) / len(text_qs))
        if result.power >= 0.80:
            print("  z-test: " + str(len(text_qs)) + " text vs " + str(len(graph_qs)) + " graph questions.")
            print("  A drop of " + str(g) + "% or more below the text accuracy (" + str(round(100 * p1)) + "%)")
            print("  is detectable with 80% power.")
            break
