"""Test 2: same question, image vs. text - does the modality matter?  (McNemar)

Our PRIMARY test. Each type-B/C question with a faithful text version was
asked twice: figure/table as IMAGE vs. the same data written as TEXT, so
only the modality changes. H0: P(correct as image) = P(correct as text).
We use the exact (binomial) version of the test. B+C is the main result;
the separate B and C results are exploratory.
"""

from statsmodels.stats.contingency_tables import mcnemar

from helpers import load_results, mcnemar_counts, fmt_p


def run_mcnemar(rows, types, label):
    """Run McNemar's test for the questions of the given types and print it."""
    both, only_image, only_text, neither = mcnemar_counts(rows, types)
    n = both + only_image + only_text + neither
    if n < 3:
        print("  " + label + ": not enough paired questions")
        return
    table = [[both, only_image], [only_text, neither]]
    result = mcnemar(table, exact=True)
    image_right = both + only_image
    text_right = both + only_text
    print("  " + label + ": " + str(n) + " pairs")
    print("     correct as image: " + str(image_right) + "/" + str(n))
    print("     correct as text:  " + str(text_right) + "/" + str(n))
    print("     only image right: " + str(only_image) + ", only text right: " + str(only_text))
    print("     " + fmt_p(result.pvalue))


rows = load_results()

print()
print("--- 2. SAME QUESTION, IMAGE vs. TEXT (McNemar's test, paired) ---")
run_mcnemar(rows, ["B"], "Type B (figures)")
print()
run_mcnemar(rows, ["C"], "Type C (tables)")
print()
run_mcnemar(rows, ["B", "C"], "Type B+C combined (PRIMARY)")
