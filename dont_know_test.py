"""Test 5: does Gemma say E ("don't know") more often on images?  (chi-square 2x2)

E was an explicit 'don't know' option in every question. If Gemma can tell
when it cannot read a figure, E should be more frequent on image input.
H0: the E-rate is the same for image input and text input.
"""

from scipy.stats import chi2_contingency

from helpers import load_results, subset, percent, fmt_p

rows = load_results()

img = subset(rows, "modality", "screenshot")
txt = []
for r in rows:
    if r["modality"] != "screenshot":
        txt.append(r)

e_img = len(subset(img, "gemma_answer", "E"))
e_txt = len(subset(txt, "gemma_answer", "E"))

print()
print("--- 5. DOES GEMMA SAY \"DON'T KNOW\" MORE ON IMAGES? (chi-square 2x2) ---")
if len(img) == 0 or len(txt) == 0:
    print("  not enough data (need both image and text answers)")
else:
    print("  image: " + str(e_img) + "/" + str(len(img)) + " answers were E (" + percent(e_img / len(img)) + ")")
    print("  text:  " + str(e_txt) + "/" + str(len(txt)) + " answers were E (" + percent(e_txt / len(txt)) + ")")
    if e_img + e_txt == 0:
        print("  Gemma never answered E - nothing to test (a finding in itself)")
    else:
        table = [[e_img, len(img) - e_img],
                 [e_txt, len(txt) - e_txt]]
        chi2, p, dof, expected = chi2_contingency(table)
        print("  chi2 = " + str(round(chi2, 3)) + ", df = " + str(dof) + "   " + fmt_p(p))
        if expected.min() < 5:
            print("  NOTE: smallest expected cell is " + str(round(expected.min(), 1)) + " (< 5),")
            print("  so the chi-square assumption is violated - interpret with care")
