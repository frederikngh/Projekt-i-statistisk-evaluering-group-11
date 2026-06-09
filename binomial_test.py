"""Test 1: is Gemma better than random guessing?  (binomial test)

H0: accuracy = 25% (random guessing among A-D)
H1: accuracy > 25% (one-sided)
"""

from scipy.stats import binomtest

from helpers import CHANCE, load_results, subset, count_correct, acc_text, fmt_p

rows = load_results()

print()
print("--- 1. IS GEMMA BETTER THAN GUESSING? (binomial test) ---")
modalities = ["text", "text_desc", "screenshot"]
for modality in modalities:
    sub = subset(rows, "modality", modality)
    if len(sub) > 0:
        result = binomtest(count_correct(sub), len(sub), CHANCE, alternative="greater")
        print("  " + modality + ": " + acc_text(sub) + "   " + fmt_p(result.pvalue))
