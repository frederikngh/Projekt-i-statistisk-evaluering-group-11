"""apply_corrections.py - apply the fixes we found when double-checking the encodings.

Surgical raw-text edits (scoped per question) so formatting stays byte-identical
apart from the intended changes. Each edit asserts the old value is present
before changing it, so re-running a batch fails loudly instead of corrupting.

    python apply_corrections.py        # apply ALL batches (fresh encodings only)
    python apply_corrections.py 1      # batch 1 only
    python apply_corrections.py 2      # batch 2 only

Batch 1 (found by re-checking every exam against its solution PDF):
  Fall2024  Q12 answer A -> B   (k-means cost = 50 = option B; worked solution says B)
  Fall2024  Q14 answer A -> D   (support(X) = 3/8 = option D)
  Fall2019  Q3  stem  x1=PM2.5 -> x1=MONTH   (x1 is MONTH per Table 1; PM2.5 is x2)
  Spring2023 Q7  figure ref "Figure 3" -> "Figure 2"  (exam numbering, not solution's)
  Spring2023 Q13 figure ref "Figure 4" -> "Figure 3"  (exam numbering)

Batch 2 (2026-06-10, found while reviewing Gemma's answers): some stems depend on
material that only appears in ANOTHER question ("the matrix V given in Question 2",
"the previous question", a figure that is not attached in the text modality).
Every question is sent to Gemma as an isolated prompt, so that material was
missing and the question unanswerable - Gemma answered E on exactly these.
The fixes inline the missing context; matrices are copied verbatim (programmatically)
from the sibling question's stem, which the exam PDFs confirm:
  Fall2017   Q24 inline the ridge-regression setup from Q23 (54 obs, 20 lambdas, LOO)
  Fall2019   Q3  inline the V matrix from Q2
  Fall2020   Q6  drop dangling "described in Question 5" (V is already inline)
  Fall2020   Q7  inline the V matrix from Q5 (the screenshot shows only scatter plots)
  Spring2018 Q12 inline the split definition from Q11 (32/24 -> 23/8 and 9/16)
  Spring2021 Q4  inline the V matrix from Q2
  Fall2023   Q4  inline the SVD from Q3 + attribute names per the exam's Table 1
  Spring2025 Q22 describe Figure 13 in text (standardized y_r centered at zero)

Batch 3 (2026-06-10): cosmetic follow-up to batch 2. Two stems still SAID
"from Question N" even though the data itself is already inline; reword so an
isolated prompt does not look like it is missing context:
  Spring2017 Q4  drop "(as defined in Question 3)" / "from Question 3"
  Spring2018 Q3  "matrix V in question 2" -> "matrix V below"
"""

import sys
from pathlib import Path

ENC = Path("data/encoded")


def q_span(text: str, qid: str) -> tuple[int, int]:
    """Start/end offsets of question `qid` in the raw JSON text."""
    start = text.index(f'"question_id": "{qid}"')
    nxt = text.find('"question_id":', start + 20)
    return start, (nxt if nxt != -1 else len(text))


def scoped_replace(text: str, qid: str, old: str, new: str) -> str:
    """Replace `old`->`new` only within the JSON span of question `qid`."""
    start, end = q_span(text, qid)
    span = text[start:end]
    assert old in span, f"{qid}: '{old[:60]}...' not found in its span"
    return text[:start] + span.replace(old, new) + text[end:]


def grab(text: str, qid: str, start_marker: str, end_marker: str) -> str:
    """Verbatim raw-text slice from question `qid` (start inclusive, end exclusive).

    Used to copy matrices byte-for-byte instead of retyping them.
    """
    a, b = q_span(text, qid)
    span = text[a:b]
    i = span.index(start_marker)
    j = span.index(end_marker, i)
    return span[i:j]


def patch(name: str, fn) -> None:
    p = ENC / f"{name}.json"
    p.write_text(fn(p.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"{name}: patched")


# ---------- batch 1 ----------

def fall2024(t: str) -> str:
    # answer_key block (top of file) - these substrings are unique to it
    assert t.count('"Q12": "A"') == 1 and t.count('"Q14": "A"') == 1
    t = t.replace('"Q12": "A"', '"Q12": "B"').replace('"Q14": "A"', '"Q14": "D"')
    # per-question correct_answer
    t = scoped_replace(t, "Q12", '"correct_answer": "A"', '"correct_answer": "B"')
    t = scoped_replace(t, "Q14", '"correct_answer": "A"', '"correct_answer": "D"')
    print("  Fall2024: Q12 A->B, Q14 A->D")
    return t


def fall2019(t: str) -> str:
    t = scoped_replace(t, "Q3", "x1=PM2.5", "x1=MONTH")
    print("  Fall2019: Q3 x1=PM2.5 -> x1=MONTH")
    return t


def spring2023(t: str) -> str:
    t = scoped_replace(t, "Q7", "Figure 3", "Figure 2")
    t = scoped_replace(t, "Q13", "Figure 4", "Figure 3")
    print("  Spring2023: Q7 Figure 3->2, Q13 Figure 4->3")
    return t


# ---------- batch 2: inline context that lived in sibling questions ----------

def fall2017_b2(t: str) -> str:
    old = ("We will again consider the ridge regression described in the previous "
           "question. Which one")
    new = ("Using the 54 observations of the Basketball dataset we consider ridge "
           "regression (regularized least squares regression) to predict the average "
           "points scored per game (y) based on the four features (x1-x4). The model "
           "minimizes with respect to w the cost function: E(w) = sum_n (y_n - [1 x_n1 "
           "x_n2 x_n3 x_n4] w)^2 + lambda w^T w, where 1 is concatenated to the data to "
           "account for the bias term. We consider 20 different values of lambda and use "
           "leave-one-out cross-validation to quantify the performance of each of these "
           "different values of lambda; the resulting training and test RMSE curves as a "
           "function of lambda are plotted in Figure 11. Which one")
    t = scoped_replace(t, "Q24", old, new)
    print("  Fall2017: Q24 ridge setup from Q23 inlined")
    return t


def fall2019_b2(t: str) -> str:
    v = grab(t, "Q2", "V = [[0.1, -0.45", "]]") + "]]"
    old = ("Consider again the PCA analysis for the Beijing air pollution dataset, in "
           "particular the SVD decomposition of X_tilde in Equation (1) (the matrix V "
           "given in Question 2, with attributes x1=MONTH, x3=PM10, x5=CO, x8=PRES, "
           "x10=RAIN, x11=WSPM used for the PCA).")
    new = ("Consider a PCA analysis of the Beijing air pollution dataset based on the "
           "standardized attributes x1=MONTH, x3=PM10, x5=CO, x8=PRES, x10=RAIN, "
           "x11=WSPM. A singular value decomposition U S V^T = X_tilde of the "
           "standardized data matrix gives\\n" + v + ",\\nwhere the rows of V correspond "
           "to the attributes in the order listed above.")
    t = scoped_replace(t, "Q3", old, new)
    print("  Fall2019: Q3 V matrix from Q2 inlined")
    return t


def fall2020_b2(t: str) -> str:
    # Q6: V is already inline; just remove the dangling cross-reference
    t = scoped_replace(
        t, "Q6",
        "Consider again the principal component analysis described in Question 5, with",
        "Consider a principal component analysis (PCA) carried out on the standardized "
        "Palmer Penguins dataset, with")
    # Q7: projecting the new point requires V, which only Q5 contained
    v = grab(t, "Q5", "V =\\n[ 0.45", "\\n\\nS =")
    old = ("Based on the principal component analysis described in Question 5, Figure 4 "
           "shows 2D scatter plots")
    new = ("A principal component analysis (PCA) is carried out on the standardized "
           "Palmer Penguins dataset (attributes x1 Bill length, x2 Bill depth, x3 "
           "Flipper length, x4 Body mass), where the singular value decomposition of "
           "the standardized data matrix gives\\n\\n" + v + "\\n\\nand the d'th column "
           "of V defines the d'th principal component direction. Figure 4 shows 2D "
           "scatter plots")
    t = scoped_replace(t, "Q7", old, new)
    print("  Fall2020: Q6 dangling ref removed; Q7 V matrix from Q5 inlined")
    return t


def spring2018_b2(t: str) -> str:
    old = ("We will consider the decision tree given by having only the above split "
           "(defined in question 11) as a decision")
    new = ("We consider a dataset of 56 airline companies, of which 32 are labeled safe "
           "(the positive class) and 24 unsafe (the negative class). A decision tree "
           "splits the data at the root according to the median value of the number of "
           "incidences (x5): the branch with relatively few incidences contains 23 safe "
           "and 8 unsafe companies, while the branch with relatively many incidences "
           "contains 9 safe and 16 unsafe companies. We will consider the decision tree "
           "given by having only this split as a decision")
    t = scoped_replace(t, "Q12", old, new)
    print("  Spring2018: Q12 split definition from Q11 inlined")
    return t


def spring2021_b2(t: str) -> str:
    v = grab(t, "Q2", "V = [[0.11, -0.8", "]]") + "]]"
    old = ("Consider again the Bicycle rental dataset and the PCA decomposition "
           "described in Equation (1) (V given in Question 2). Recall the PCA "
           "decomposition is obtained")
    new = ("Consider the Bicycle rental dataset and the PCA decomposition "
           "U S V^T = X_tilde, where\\n" + v + "\\nand the columns of V are the "
           "principal component directions. Recall the PCA decomposition is obtained")
    t = scoped_replace(t, "Q4", old, new)
    print("  Spring2021: Q4 V matrix from Q2 inlined")
    return t


def fall2023_b2(t: str) -> str:
    v = grab(t, "Q3", "V ~ [[0.43", "]]") + "]]"
    old = "We consider again the PCA result of the CCPP dataset."
    new = ("We consider the PCA of a subset of the CCPP dataset, where the SVD of the "
           "centered data matrix gives the singular values Sigma ~ diag(3.7, 3.04, "
           "0.56, 0.48) (the remaining singular value is 0) and\\n" + v + ",\\nwhere "
           "the rows of V correspond to the attributes x1 (Temperature), x2 (Pressure), "
           "x3 (Humidity), x4 (Vacuum).")
    t = scoped_replace(t, "Q4", old, new)
    print("  Fall2023: Q4 SVD from Q3 inlined (attribute names per the exam's Table 1)")
    return t


def spring2025_b2(t: str) -> str:
    old = "Figure 13 shows the distribution of the standardized y_r attribute to be predicted."
    new = ("The standardized y_r attribute to be predicted is distributed around zero "
           "and takes both negative and positive values.")
    t = scoped_replace(t, "Q22", old, new)
    print("  Spring2025: Q22 Figure 13 described in text")
    return t


# ---------- batch 3: cosmetic cross-reference cleanup (data already inline) ----------

def spring2017_b3(t: str) -> str:
    t = scoped_replace(
        t, "Q4",
        "The data projected onto the two first principal components (as defined in "
        "Question 3) is given in Figure 2",
        "The data projected onto the two first principal components is given in "
        "Figure 2")
    t = scoped_replace(t, "Q4", "The V matrix from Question 3 is:",
                       "The V matrix of principal component directions "
                       "(columns are the components) is:")
    print("  Spring2017: Q4 cross-question mentions reworded")
    return t


def spring2018_b3(t: str) -> str:
    t = scoped_replace(
        t, "Q3",
        "According to the extracted PCA directions given by the matrix V in "
        "question 2 what will be the coordinate",
        "According to the extracted PCA directions given by the matrix V below, "
        "what will be the coordinate")
    print("  Spring2018: Q3 cross-question mention reworded")
    return t


BATCH1 = [("Fall2024", fall2024), ("Fall2019", fall2019), ("Spring2023", spring2023)]
BATCH2 = [("Fall2017", fall2017_b2), ("Fall2019", fall2019_b2), ("Fall2020", fall2020_b2),
          ("Spring2018", spring2018_b2), ("Spring2021", spring2021_b2),
          ("Fall2023", fall2023_b2), ("Spring2025", spring2025_b2)]
BATCH3 = [("Spring2017", spring2017_b3), ("Spring2018", spring2018_b3)]

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    batches = {"1": BATCH1, "2": BATCH2, "3": BATCH3,
               "all": BATCH1 + BATCH2 + BATCH3}[which]
    for name, fn in batches:
        patch(name, fn)
    print(f"All corrections applied (batch: {which}).")
