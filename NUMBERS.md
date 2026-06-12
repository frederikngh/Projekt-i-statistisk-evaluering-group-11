# Numbers for the statistical tests

Source: `python summarize.py` on the final `data/results.csv` (532 rows, collection
finished 2026-06-11). One row per (question, input format); correct = Gemma's letter
matches the official answer key.

Question types, written out:

- **Pure-text questions** (type A) — answerable from words alone, no figure or table
  (134 questions).
- **Figure questions** (type B) — need a graph/plot to answer (157 total: 13 are simple
  enough to write out faithfully as text, 144 are geometric — scatter plots, dendrograms,
  ROC curves, contours — and exist only as images).
- **Table questions** (type C) — need a table/matrix of numbers (114 questions; all can
  be written out as text).

Input formats: `text` = a pure-text question as text; `text_desc` = the figure/table
written out as text; `screenshot` = the figure/table as a cropped image (stem and
options always given as text in both arms — only the figure's format changes).

How to compute each test: see `STATISTICS.md` (same section numbers).

---

## 1. Binomial tests vs. chance (one-sided, p0 = 0.25), one per input format

| how the question was given          | correct | n   | accuracy |
|-------------------------------------|---------|-----|----------|
| pure-text question, as text         | 98      | 134 | 73.1%    |
| figure/table written out as text    | 80      | 127 | 63.0%    |
| figure/table as an image            | 106     | 271 | 39.1%    |

## 2. McNemar, exact (THE primary test) - the 127 figure/table questions asked both ways

|                         | text version right | text version wrong | total |
|-------------------------|--------------------|--------------------|-------|
| **image version right** | 45                 | 9                  | 54    |
| **image version wrong** | 35                 | 38                 | 73    |
| **total**               | 80                 | 47                 | 127   |

Disagreements (the only pairs the test uses): 9 favoured the image, 35 favoured the
text, 44 in total.

Descriptive subgroup breakdown (NOT tested - see STATISTICS.md section 4 "Why no
per-type tests"): among the 114 table questions the disagreements split 29 text vs.
9 image; among the 13 writable figure questions, 6 text vs. 0 image.

## 3. Two-proportion z-test (unpaired): pure-text vs. geometric figure questions

| group                                          | correct | n   | accuracy |
|------------------------------------------------|---------|-----|----------|
| pure-text questions, asked as text             | 98      | 134 | 73.1%    |
| geometric figure questions (image-only)        | 52      | 144 | 36.1%    |

Pooled proportion for the z statistic: (98 + 52) / (134 + 144) = 150/278 = 0.540.

Caveat to state: different questions in the two groups, so question difficulty is
confounded with type - descriptive, not causal.

## 4. Chi-square (2x3, df = 2): do the question types differ, all given as text?

|         | pure-text questions | figure questions (the 13 writable ones) | table questions |
|---------|---------------------|------------------------------------------|-----------------|
| correct | 98                  | 11                                       | 69              |
| wrong   | 36                  | 2                                        | 45              |
| total   | 134                 | 13                                       | 114             |

Caveats to state: (1) the figure column is only the 13 text-faithful figure questions,
not a representative sample of figure questions (selection bias); (2) its expected
"wrong" count is 13 x 83/261 = 4.1, just under the "expected counts >= 5" rule of
thumb - flag this in the assumption check.

## 5. Chi-square (2x2, df = 1): does Gemma answer E ("don't know") more on images?

|                            | answered E | answered A-D | total |
|----------------------------|------------|--------------|-------|
| question given as an image | 10         | 261          | 271   |
| question given as text     | 5          | 256          | 261   |

## 6. Contamination check: the 15 old exams vs. the held-out Fall 2025 exam

Two-proportion z-test, ONE-SIDED (contamination predicts old > clean). The 15 old
exams are public PDFs that may be in Gemma's training data; Fall 2025 postdates the
model's training data, so it is the one exam Gemma cannot have seen.

| group                                  | correct | n   | accuracy |
|----------------------------------------|---------|-----|----------|
| 15 old exams (all 532 rows)            | 284     | 532 | 53.4%    |
| Fall 2025 (held out)                   | PENDING | 34  | PENDING  |

The Fall 2025 rows are collected separately (on the PC, same model + protocol) and
do NOT enter any of the analyses above - sections 1-5 stay defined on the 15 old
exams only.

Descriptive context (accuracy per old exam - relevant because one clean exam means
exam difficulty is confounded; the old exams vary this much on their own):

| exam       | accuracy | exam     | accuracy |
|------------|----------|----------|----------|
| Spring2017 | 44.4%    | Fall2017 | 36.1%    |
| Spring2018 | 42.9%    | Fall2018 | 55.3%    |
| Spring2019 | 51.3%    | Fall2019 | 69.2%    |
| Spring2020 | 46.2%    | Fall2020 | 66.7%    |
| Spring2021 | 61.3%    | Fall2021 | 63.2%    |
| Spring2023 | 51.5%    | Fall2023 | 54.3%    |
| Spring2024 | 54.5%    | Fall2024 | 40.0%    |
| Spring2025 | 61.8%    |          |          |

Note: the two OLDEST exams (longest on the web) score LOWEST - the opposite of what
training-data exposure would predict. Useful descriptive argument in the report.

## 7. Power check inputs

- McNemar: 44 disagreements - find the smallest number of same-direction
  disagreements out of 44 whose exact two-sided p is below 0.05.
- Two-proportion z: group sizes 134 and 144.
