# Statistics in this project — tests, theory, assumptions, and where the materials use them

Reference for writing the report's methods/results sections and for the individual oral.
The project description (step 5b) grades exactly this: *"Sufficiently describe why the
statistical test is relevant, what the assumptions for the test are, and if they are
completely/approximately met."* Every section below answers those three questions for one
of our tests, and points to where the same method appears in the course materials
(`materials/` in the parent folder; page numbers are the printed page numbers).

The six analysis scripts this file documents:

| # | Script | Question it answers | Test | H0 | Null distribution |
|---|--------|--------------------|------|----|-------------------|
| 1 | `binomial_test.py` | Is Gemma better than guessing (per modality)? | Exact binomial, one-sided | accuracy = 25 % | Binomial(n, 0.25) |
| 2 | `mcnemar_test.py` | **PRIMARY:** same question, image vs. text — does modality matter? | McNemar, exact (paired) | P(correct as image) = P(correct as text) | Binomial(n_d, ½) on discordant pairs |
| 3 | `text_vs_graph_test.py` | Pure-text questions vs. pure-graph questions | Two-proportion z (unpaired) | p_text = p_graph | N(0,1) |
| 4 | `question_types_test.py` | Does accuracy differ across types A/B/C (all in text form)? | Chi-square, 3×2 | all types equally accurate | χ² with df = 2 |
| 5 | `dont_know_test.py` | Does Gemma say E ("don't know") more on images? | Chi-square, 2×2 | same E-rate both inputs | χ² with df = 1 |
| 6 | `power_check.py` | How big must an effect be before we can see it? | Power analysis (not a test) | — | — |

Plus, everywhere: **95 % Wilson confidence intervals** on every reported accuracy
(`helpers.py: wilson_ci`), and the significance level **α = 0.05** (`fmt_p`).

---

## 0. The probability model under everything

Every row in `results.csv` is one prompt with a binary outcome: correct (1) or wrong (0).
This is the classic test-set framing from Raschka §1.7 (pp. 10–11): each answered question
is a **Bernoulli trial**, so the number of correct answers out of n is

$$X \sim \text{Binomial}(n, p), \qquad P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

with p = the accuracy we want to learn about. The same binomial distribution is on the
stats cheat sheet (p. 6). Everything below — binomial test, Wilson CI, z-test, chi-square,
McNemar — is this one model plus different comparisons.

Two things about *our* setup are worth saying precisely at the oral:

- **Where the randomness lives.** `collect.py` decodes greedily (`do_sample=False`), so
  Gemma's answer to a fixed prompt is deterministic and reproducible. The random quantity
  is therefore *which questions we have*: we treat the 405 exam questions as a sample from
  the (hypothetical) population of 02450-style MCQs, and p is Gemma's accuracy on that
  population. This is the standard framing for evaluating a fixed model on a test set
  (Raschka §1.3–1.7). We use *all* public exams 2017–2025, so strictly the inference target
  is the superpopulation of similar questions, not some larger pool we subsampled — say
  this honestly in the report.
- **Independence between trials.** Each prompt is a separate, stateless model call with no
  chat history, and the model weights are frozen. This is exactly the precaution the
  exemplar paper took ("For every prompt a new chat was opened to ensure no dependence on
  previous prompts", LLM_bias_1_revision.pdf, App. C.1, p. 17) and what the project
  description's step 3 worries about for online tools (which learn from your prompts —
  a local model removes that problem entirely). Residual caveat: questions from the same
  exam share topics, so within-exam correlation can't be fully excluded; see §8.

Chance level is **25 %**: every question has exactly four scoreable options A–D (E is
never the keyed answer). If Gemma guessed uniformly over all five letters its accuracy
would be 20 % < 25 %, so testing against 25 % is the *conservative* choice — it can only
make "better than guessing" harder to claim.

---

## 1. Accuracy with Wilson confidence intervals (`helpers.py`)

**Purpose.** Step 5b explicitly asks us to "present the stability (uncertainty/variance)
of the performance". A point accuracy of, say, 76.9 % (20/26) means little without an
interval; we attach a 95 % CI to every accuracy we print.

**Theory.** The textbook interval is the normal approximation (Wald) interval that
Raschka derives in §1.7 (eq. 10–17, pp. 10–11):
$\hat p \pm z\sqrt{\hat p(1-\hat p)/n}$ with z = 1.96. It behaves badly for small n or
$\hat p$ near 0 or 1 (it can extend past [0,1] and has poor coverage). The **Wilson (1927)
interval** fixes this by solving the score test inversion; centre and half-width are

$$\frac{\hat p + z^2/2n}{1 + z^2/n} \;\pm\; \frac{z}{1 + z^2/n}\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}.$$

It is pulled slightly toward ½ and always stays inside [0,1].

**Assumptions.** Independent Bernoulli trials with common p (the §0 model). No normality
assumption on the data itself — the approximation is on the *binomial proportion*, and
Wilson is precisely the variant that stays accurate at small n.

**Where the materials use it.** The cross-validation paper by Bayle et al. (NeurIPS 2020,
`Cross_Validation_Confidence_Intervals_For_Test_Error.pdf`, §5, p. 7) uses exactly this:
"95 % Wilson intervals, which are known to provide more accurate coverage for binomial
proportions than a ±2 standard error interval", citing Wilson (1927) and Brown, Cai &
DasGupta (2001). Raschka §1.7 gives the Wald interval that Wilson improves on.

**Implementation.** `statsmodels.stats.proportion.proportion_confint(k, n, method="wilson")`.

**Caveat to state.** The Wilson CIs printed next to the McNemar result are *marginal*
(they treat the image-arm and text-arm accuracies as separate samples and ignore the
pairing). Fine as descriptives; a CI on the paired *difference* would need a paired method
(e.g. bootstrap over pairs — see §9).

---

## 2. Test 1 — exact binomial test against chance (`binomial_test.py`)

**Purpose.** Manipulation check before anything comparative: per modality (text,
text_desc, screenshot), is Gemma doing better than random guessing? If a modality is at
chance, comparative results for it mean something very different.

**Hypotheses.** H0: p = 0.25. H1: p > 0.25, **one-sided** — decided a priori because
"above chance" is the only direction that changes our interpretation (see §8 on
sidedness).

**Theory.** Under H0 the correct-count is exactly Binomial(n, 0.25) — see §0; the p-value
is the exact upper tail sum $P(X \ge k_{obs}) = \sum_{i=k_{obs}}^{n}\binom{n}{i}0.25^i\,0.75^{\,n-i}$.
No approximation, valid at any n. A nice property: under *guessing*, every question has
success probability exactly 0.25 regardless of how hard it is, so heterogeneous question
difficulty does not break the null distribution — difficulty only matters under H1
(where it affects power, not validity).

**Assumptions.**
1. Binary outcome per trial — holds by construction (answer matches key or not).
2. Fixed number of trials n — holds (the question set is fixed before collection).
3. Independent trials — argued in §0; the within-exam clustering caveat applies (§8).
4. Same null success probability 0.25 each trial — holds under H0 by the guessing argument
   above (and 25 % is conservative given the E option).

**Where the materials use it.** The binomial distribution and its tail computation are on
the cheat sheet (p. 6). Raschka builds the same "correct predictions ~ binomial" model in
§1.7 (p. 10) and uses **exact binomial p-values** in §4.4 (pp. 37–38) — there as the exact
version of McNemar, i.e. literally the computation our test 2 reuses.

**Implementation.** `scipy.stats.binomtest(k, n, 0.25, alternative="greater")`.

---

## 3. Test 2 (PRIMARY) — McNemar's exact test, paired (`mcnemar_test.py`)

**Purpose.** The core of the project. Each type-B/C question with a faithful text version
was asked twice: figure/table as a cropped **image** vs. the same information written as
**text**, with the stem and options identical in both arms (design B). The *only* thing
that changes is the modality, so a paired comparison isolates the graph/table-reading
penalty from question difficulty.

**Hypotheses.** H0: P(correct as image) = P(correct as text) for the paired questions.
H1: the probabilities differ (two-sided).

**Theory.** Cross-tabulate each pair (Raschka Fig. 19, p. 36):

|                | text correct | text wrong |
|----------------|--------------|------------|
| **image correct** | a (both)   | b (only image) |
| **image wrong**   | c (only text) | d (neither) |

The pairs where both arms agree (a and d) tell us nothing about a *difference* — a
question so easy both arms get it, or so hard both miss it, is consistent with any H0/H1.
The information sits in the **discordant pairs** n_d = b + c. Under H0, each discordant
pair is equally likely to favour either modality, so

$$b \mid n_d \sim \text{Binomial}(n_d, \tfrac12),$$

and the exact two-sided p-value is $p = 2\sum_{i=b}^{n_d}\binom{n_d}{i}0.5^{n_d}$ (for
b > c; Raschka §4.4, p. 38). This conditioning-on-disagreement is precisely what removes
per-question difficulty as a confounder — the strongest sentence to say at the oral.

The large-sample version is McNemar's chi-square (Raschka §4.3, p. 37):
$\chi^2 = (b-c)^2/(b+c) \sim \chi^2_1$, which is just the squared, standardized binomial:
$(b - n_d/2)/\sqrt{n_d/4}$ squared. Edwards' continuity-corrected variant is
$(|b-c|-1)^2/(b+c)$. Raschka's rule of thumb: the χ² approximation is reasonable when
b + c > 25, and the corrected version matches the exact test well when b, c > 50. Our
discordant counts may be small, **which is why we run the exact version** —
`mcnemar(table, exact=True)` — and never have to defend an approximation.

**Assumptions.**
1. **Pairs are independent of each other** — each pair is one exam question, answered in
   two stateless calls (§0).
2. **Within a pair, the two arms answer the same question** — guaranteed by design B: the
   stem and options are byte-identical, only figure-as-image vs. figure-as-text differs.
   (This was audited: an early encoding bug had table data embedded in ~70 stems, sending
   it to BOTH arms and nullifying the contrast; fixed in corrections batch 4.)
3. The two arms of a pair MAY be dependent — that is the point of pairing, not a violation.
4. No order/carry-over effects — holds because the arms are separate calls with no shared
   state (a human answering twice would have a memory confound; the model does not).

**Multiple testing.** We run McNemar three times (B only, C only, B+C — nested subsets).
**B+C is declared the primary confirmatory test up front; the per-type runs are
exploratory.** This pre-declaration is the clean alternative to a Bonferroni correction
(α/m) and follows Raschka §4.5 (pp. 38–39): either control the family-wise error or
clearly separate confirmatory from exploratory analyses.

**Where the materials use it.** Raschka devotes §4.3–4.4 (pp. 35–38) to McNemar for
comparing two classifiers on the same items, and reports Dietterich's (1998) simulation
summary (p. 43): McNemar has a **low false-positive rate** and "is a good choice if the
model fitting can only be conducted once" — our situation exactly (one frozen model, no
retraining). The Bayle et al. CV paper mentions the same Dietterich comparison (footnote
6, p. 7). McNemar is the paired-nominal analogue of the cheat sheet's **paired t-test**
(p. 2: "subtract the two values for each individual… ie before and after values for each
individual patient" — replace patient with question, before/after with text/image). Its
generalization to ≥3 conditions is Cochran's Q (Raschka §4.6, pp. 39–40) — we only have
two conditions, so McNemar suffices.

**Implementation.** `statsmodels.stats.contingency_tables.mcnemar(table, exact=True)`.

**Honest weakness.** The paired set is 114 type-C (tables) + only 13 type-B (figures), so
the primary result is mostly a *table*-reading penalty; the B-only McNemar is underpowered.
The pure-graph effect lives in test 3 — with its own, different weakness.

---

## 4. Test 3 — two-proportion z-test, unpaired (`text_vs_graph_test.py`)

**Purpose.** Geometric figures (scatter plots, dendrograms, ROC curves, contours…) have no
faithful text form, so they cannot enter the paired design. We compare type-A pure-text
questions (n = 134) against these screenshot-only type-B questions (n = 144) as two
independent groups.

**Hypotheses.** H0: p_text = p_graph. H1: they differ (two-sided).

**Theory.** With independent groups, $\hat p_1 - \hat p_2$ is approximately normal for
moderate n (CLT applied to the binomial counts of §0). Under H0 both groups share one p,
estimated by pooling $\hat p = (k_1+k_2)/(n_1+n_2)$, giving

$$z = \frac{\hat p_1 - \hat p_2}{\sqrt{\hat p(1-\hat p)\left(\frac{1}{n_1}+\frac{1}{n_2}\right)}} \;\sim\; N(0,1) \text{ under } H_0.$$

On a 2×2 table this z-test and the chi-square test are the same test: **z² = χ²** (the
uncorrected χ²) — our defence line for using a z-test when the project description's
step 5b lists "Chi-square test (for proportions)". It is also the proportions analogue of
the cheat sheet's **unpaired two-sample t-test** (pp. 2–3): same logic of
difference-over-pooled-standard-error, with the binomial supplying the variance.

**Assumptions.**
1. Independent observations within and *between* groups — holds: the two groups are
   disjoint question sets, every item answered once. (Note: Raschka §4.2 (pp. 34–35) and
   Dietterich criticize this test heavily — but for the scenario of *two models on the
   same test set*, where the two proportions are computed on identical items and are
   therefore dependent, inflating false positives. That specific violation does not apply
   to us: one model, two disjoint item sets. Worth pre-empting at the oral.)
2. Normal approximation adequacy — rule of thumb: expected successes and failures
   ≥ ~5–10 per group; with n = 134 and 144 this holds unless accuracy is extreme (check
   against the observed counts when the data is in).
3. Common p within each group under H0 — same superpopulation framing as §0.

**The unavoidable caveat (say it, don't hide it).** Unlike test 2, the two groups contain
*different questions*: question difficulty and topic are confounded with modality. A
significant difference shows "Gemma is worse on these graph questions than on these text
questions", not "the image rendering causes the gap". That causal claim is what the paired
McNemar is for; this test provides the dramatic descriptive contrast on the question types
where pairing is impossible. The docstring and report both state this.

**Where the materials use it.** Raschka §4.2 (the test itself + its limits); project
description step 5b (chi-square for proportions = the same test, see z² = χ²); slides
part 1, slide 18, list chi-square for exactly this kind of categorical comparison.

**Implementation.** `statsmodels.stats.proportion.proportions_ztest([k1,k2], [n1,n2])`
(pooled, no continuity correction — hence exactly z² = uncorrected χ²).

---

## 5. Test 4 — chi-square test of homogeneity across question types (`question_types_test.py`)

**Purpose.** With modality held fixed at *text* (type A as plain text, B/C as their text
descriptions), does accuracy depend on question *type*? This separates "the content type
is hard" from "the image rendering is hard".

**Hypotheses.** H0: p_A = p_B = p_C. H1: at least one differs.

**Theory.** Build the 3×2 table of (correct, wrong) counts per type. Expected counts under
independence: $E_{ij} = (\text{row}_i \times \text{col}_j)/\text{total}$, statistic
$\chi^2 = \sum (O-E)^2/E$ with df = (r−1)(c−1) = 2. This is the cheat sheet's chi-square
test of independence verbatim (pp. 4–5, including the expected-count formula and the
fitness/smoking worked example); the χ² distribution arises as the sum of squared
approximately-standard-normal deviations of the cell counts.

**Assumptions.**
1. Independent observations, each in exactly one cell — holds (each row of the table is
   one question answered once; types are mutually exclusive).
2. **All expected counts ≥ 5** — the classical validity rule (cheat sheet p. 4). The
   script computes `expected.min()` and prints a warning when violated. With only 13
   type-B items this is the cell to watch; if it fails, report the warning and interpret
   descriptively (or merge B into B+C).
3. Categories fixed in advance — holds (A/B/C assigned at encoding time, before any model
   run).

**Omnibus logic.** Chi-square over three groups only says *that* types differ, not which
pair differs — the same omnibus-then-post-hoc structure as ANOVA (Raschka §4.5,
pp. 38–39: run the omnibus test first; only if it rejects, do pairwise tests with a
Bonferroni correction). If we ever need the pairwise follow-up: three 2×2 chi-squares at
α/3.

**Selection-bias caveat.** "Type B" here means only the 13 text-faithful type-B items —
the describable minority, not representative of figures in general. State it.

**Where the materials use it.** Cheat sheet pp. 3–5 (both goodness-of-fit and
independence variants); slides part 1, slide 18: chi-square = "checking if the performance
of an AI model is independent of the dataset used" — our question with "dataset" replaced
by "question type". The exemplar paper's two-way ANOVA (§3.2, p. 8) plays this exact
omnibus role for a *continuous* outcome (number of STEM suggestions, gender × age); ours
is the categorical-outcome counterpart.

**Implementation.** `scipy.stats.chi2_contingency(table)`; df = 2 here, so no Yates
correction is involved (scipy only applies it to 2×2 tables).

---

## 6. Test 5 — E-rate ("don't know") chi-square, 2×2 (`dont_know_test.py`)

**Purpose.** Every prompt offers E = "don't know". If Gemma has any awareness of when it
cannot read a figure, E should be more frequent on image input than on text input. This is
our taught-statistics substitute for confidence calibration (which we descoped — §9): it
measures *expressed* uncertainty behaviourally instead of reading internal probabilities.

**Hypotheses.** H0: P(answer = E | image input) = P(answer = E | text input). Two-sided.

**Theory.** 2×2 table: rows = input kind (screenshot vs. text/text_desc), columns =
(answered E, answered A–D). Same chi-square machinery as test 4 with df = 1. Note the
outcome here is "said E", not "correct" — a different Bernoulli variable on the same calls.

**Assumptions.** As in test 4: independent observations, expected counts ≥ 5. The
expected-count rule matters *more* here because E may be rare (in the no-CoT pilot Gemma
said E mostly on broken questions; possibly never after the fixes). The script:
- prints "Gemma never answered E — nothing to test (a finding in itself)" when there are
  zero E's overall: a model that *never* admits uncertainty on unreadable figures is a
  reportable behavioural result, no p-value needed;
- prints the expected-count warning otherwise. If expected counts are < 5, the standard
  remedy to name at the oral is **Fisher's exact test** (`scipy.stats.fisher_exact`),
  which conditions on the margins and is valid at any counts.

**Implementation detail worth knowing.** `scipy.stats.chi2_contingency` applies **Yates'
continuity correction by default on 2×2 tables** (df = 1), so the printed χ² here is the
corrected one — slightly conservative, appropriate at small counts. (Same family of
correction as Edwards' for McNemar, Raschka p. 37.)

**Where the materials use it.** Chi-square anchors as in test 4 (cheat sheet pp. 4–5,
slides part 1 slide 18). The LLM-evaluation survey describes refusal-rate as an
established metric: "the metric … is the proportion of cases where LLMs refuse to answer"
(LLM_evalaution_survey.pdf, p. 22, on the TrustGPT PVA evaluation) — our E-rate is that
metric with an explicit opt-out option.

---

## 7. Power check — minimum detectable effects (`power_check.py`)

**Purpose.** Not a hypothesis test: a *design* justification. Step 4d of the project
description asks "How are you deciding on the number of times you will prompt the GenAI?
(Hint: sample size estimation, if applicable.)" Our item budget is fixed by reality (15
public exams × 27 questions, of which 127 are pairable and 134/144 form the unpaired
groups), so instead of choosing n for a target effect, we answer the inverse question:
**given our n, how big must the true effect be before our tests would reliably detect
it?** Both directions are standard power analysis.

**Concepts** (cheat sheet p. 5): Type I error = rejecting a true H0, probability α = 0.05;
Type II error = failing to reject a false H0, probability β; **power = 1 − β**, with 80 %
the usual convention.

**Part 1 — McNemar.** The exact McNemar test only uses the n_d discordant pairs (§3), so
its sensitivity is fully described by: out of n_d disagreements, how many must favour the
same modality before the exact binomial p drops below 0.05? The script scans k = ⌈n_d/2⌉+1
… n_d and reports the first k that rejects (e.g. on the synthetic example data: 16
disagreements → at least 13 must point the same way). If n_d is tiny, it reports that
significance is impossible — the honest statement that the B-only analysis is
underpowered.

**Part 2 — two-proportion z.** Using the normal-approximation power function
(`statsmodels.stats.proportion.power_proportions_2indep`) with our actual group sizes
(134 vs. 144), the script scans accuracy drops of 1 %, 2 %, … below the observed text
accuracy and reports the smallest drop detectable with ≥ 80 % power. That number is the
"minimum detectable effect" to quote in the report.

**Where the materials use it.** The exemplar paper is the template (§3.2, p. 8): a pilot
study (2 prompts × 10 iterations) → pooled SD 1.07 → **Cohen's d = 1.1** → "Power t-test:
p=0.8, sd=1.07, α=0.05, d=0.5 → n = 74", rounded to 80 per group. Ours is the same
calculation run in the other direction, for proportions instead of means, because our n
is externally fixed. Raschka §4.13 (p. 45) adds the complementary warning: with large
samples *everything* becomes significant, so report effect sizes (our accuracy
differences with CIs), not just p-values — statistical significance ≠ practical
significance. The Bayle et al. paper evaluates competing tests *by* their power curves
(Fig. 2, p. 8) — power as a first-class design criterion.

---

## 8. Cross-cutting assumptions and choices (the oral lives here)

**Independence — the one assumption every test shares.** Our three structural arguments:
(1) stateless, history-free model calls (the local-model version of the exemplar's
new-chat-per-prompt and of project-description step 3's warning about online tools);
(2) frozen weights — no learning between calls; (3) one answer per (question, modality),
no resampling, so no reuse of items within a test. Remaining honest caveat: the 27
questions of one exam share topics and sub-figures, so observations are not perfectly
independent *across* items of an exam. The paired McNemar is robust to this (differencing
within a question removes shared question effects; pairs remain the independent units).
For the unpaired tests, clustering would make the effective n somewhat smaller than the
nominal n — mention as a limitation rather than modelling it (cluster-robust methods are
beyond the course's scope, and beyond what the data can support).

**Why proportion tests instead of t-tests/ANOVA.** Our outcome is 0/1. The mean of 0/1
data *is* a proportion, its variance is determined by the mean (p(1−p)), and normality of
residuals can never hold — the situations the slides flag as "assumptions not met"
(slide 19). Rather than transforming (nothing to transform) or going rank-based, the
correct move for nominal data is the binomial/chi-square family, which models the 0/1
outcome exactly. McNemar is literally the non-parametric paired test for nominal data
(Raschka p. 35 calls it "a non-parametric statistical test for paired comparisons"). The
exemplar paper used two-way ANOVA + Box-Cox because its outcome was a *count* 0–10 per
answer (§3.2) — different outcome type, hence different test family. Being able to say
"we matched the test family to the outcome type" is the whole game.

**One-sided vs. two-sided.** Binomial test: one-sided (H1: better than chance), fixed a
priori — below-chance performance has no separate scientific story here, and the
directional question is the only one we'd act on. McNemar and z-test: two-sided — we
*expect* images to be worse, but the tests stay agnostic; a two-sided test is conservative
and avoids any suspicion of choosing the direction after seeing data. (The cheat sheet
p. 2 explains exactly this halving/doubling of tail probability.)

**Multiple testing across the analysis.** One pre-declared primary confirmatory result:
**McNemar on B+C**. Everything else (per-type McNemars, the unpaired z, the type
chi-square, the E-rate) is supporting/exploratory and is reported with that label rather
than with a correction. If an examiner wants a correction anyway: Bonferroni α/m
(Raschka §4.5, pp. 38–41, including Perneger's caveat that Bonferroni buys Type-I control
at the cost of Type-II errors).

**p-values AND intervals.** Every accuracy is reported with its Wilson CI, every test
with its p-value (formatted by `fmt_p`: "p < 0.001" or rounded to 3 decimals, plus the
α = 0.05 verdict). The CI answers "how big and how uncertain", the p-value answers "is it
distinguishable from H0" — step 5b asks for both.

**Training-data contamination.** The exams are public PDFs from 2017–2025; Gemma may have
seen them in pre-training. This would inflate *absolute* accuracy (test 1) and is named in
the report. The paired *contrast* (test 2) is partially protected — both arms share the
identical stem, so memorization helps both arms; only the figure rendition differs — but
"partially" is the right word, since a memorized figure discussion could leak asymmetrically.

**Internal validity constants.** One model (`gemma-4-E2B-it`) for all 532 calls, one
decoding setting (greedy, same token budget), one prompt template per modality. Any
between-arm difference is therefore attributable to the input rendition, not to runtime
configuration drift.

---

## 9. Methods in the materials we deliberately do NOT use — and why

Likely oral questions ("you read about X — why isn't it in your analysis?"):

- **Pearson / Spearman correlation** (slides part 1, slides 6–17, with the P/A
  purpose-assumption format). Measures association between two *paired numeric* variables.
  We have a binary outcome vs. categorical factors — nothing to correlate. (If we had kept
  model confidence, correct-vs-confidence would have been its use case; descoped.)
- **t-tests / ANOVA** (cheat sheet pp. 2–3; slides slide 18; exemplar §3.2). Right tool
  for continuous/count outcomes; wrong distribution family for 0/1 — see §8. The
  exemplar's ANOVA is our role model for *rigor* (model equation written out, residual
  QQ-plots, Box-Cox for non-normal counts, independence by design), not for the specific
  test.
- **Non-parametric rank tests** (Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis —
  slides slide 19's fallback ladder). They serve continuous/ordinal data failing
  normality. For nominal data the chi-square/McNemar family already *is* the
  assumption-light choice.
- **Cross-validation, 5×2cv t-tests, nested CV** (Raschka §3 and §4.8–4.12;
  both CV papers). CV exists to handle the variance from *training* on resampled data and
  the dependence it induces: Bates, Hastie & Tibshirani show naive CV intervals undercover
  because fold errors are correlated (§1.1 p. 1434: nominal 10 % miscoverage became 31 %;
  §4.1 p. 1440 traces it to the error covariance), and Bayle et al. build the CLT
  machinery to repair it. **We train nothing.** One frozen model, each item evaluated
  once — there is no fold structure, no refitting variance, and the plain binomial model
  of §0 applies directly. Raschka's own summary points here (p. 43): McNemar is the
  recommended test "if the model fitting can only be conducted once". (The CV papers are
  directly relevant to the *individual* assignment instead, whose Task 2 is designing a CV
  scheme for repeated-measures data.)
- **Bootstrap confidence intervals** (Raschka §2, pp. 15–21). A legitimate alternative
  for uncertainty on accuracies; we use Wilson because it is the standard closed-form
  interval for a single proportion. The one place bootstrap would add something we don't
  have: a CI on the *paired* accuracy difference (resample the 127 pairs) — noted as an
  extension, not required for the conclusions.
- **Calibration metrics — ECE, Brier, reliability diagrams, logistic regression
  correct ~ confidence.** Descoped 2026-06-09: they require reading token probabilities,
  are not part of the taught material, and would be indefensible at an oral graded on
  02402-level statistics. The E-rate test (test 5) keeps a behavioural version of the
  uncertainty question inside the taught toolbox.
- **LLM-as-a-judge / BLEU / human rating scales** (project description step 2; the
  LLM-evaluation survey passim). Needed when there is no ground truth. Our MCQs have an
  official answer key — exact scoring, no judge required. (The exemplar paper is the
  no-ground-truth case: open-ended answers, hand-categorized, count outcome.)

---

## 10. Map of the materials

| Material | What it contributes to our stats |
|----------|----------------------------------|
| `group assignment/02445_Project_June2026.pdf` | The grading contract: step 5b (justify test, state assumptions, check them; "present stability/uncertainty"), step 4d (sample-size estimation → our power check), step 3 (prompt independence). Lists t-test / chi-square / ANOVA as the expected level of tooling. |
| `group assignment/LLM_bias_1_revision.pdf` (exemplar, Due et al. FAccT '24) | The rigor template: model written out (two-way ANOVA, §3.2 p. 8), assumptions checked (QQ-plots, Box-Cox), independence by fresh chat per prompt (App. C.1), pilot → Cohen's d → power-based n = 80/group, results table w/ F and p (Table 2, p. 10). Our binary-outcome translation of each element: §0, §7, and the per-test assumption blocks above. |
| `stats_cheat_sheet.pdf` | The 02402 backbone: t-tests incl. the paired/unpaired distinction that motivates McNemar-vs-z (pp. 2–3), chi-square GoF + independence with expected counts and df (pp. 3–5), Type I/II errors (p. 5), the binomial distribution (p. 6), t and χ² tables (pp. 7–8). |
| `other/02445_2026_part1.pdf` (course slides) | The course's P:/A: (purpose/assumptions) presentation style we mirror here; Pearson vs. Spearman (slides 6–17); the test refresher incl. "chi-square: is performance independent of the dataset used" (slide 18); the assumptions-not-met ladder (slide 19). |
| `raschka_StatEva_modelComparison.pdf` | The ML-evaluation theory source: accuracy as binomial + normal-approx CI (§1.7 pp. 10–11), difference-of-proportions z-test and its limits (§4.2 pp. 34–35), McNemar χ², continuity correction, exact binomial form (§4.3–4.4 pp. 35–38), multiple testing/Bonferroni/omnibus-then-post-hoc (§4.5 pp. 38–41), Cochran's Q (§4.6), Dietterich's test comparison (p. 43), effect size vs. p-value (§4.13 p. 45). |
| `Cross_Validation_Confidence_Intervals_For_Test_Error.pdf` (Bayle et al., NeurIPS 2020) | Uses 95 % **Wilson intervals** for binomial proportions, citing Wilson 1927 and Brown et al. 2001 (§5 p. 7) — our CI choice, used for the same reason. Shows what valid inference takes when estimates ARE dependent (CV) — the machinery we avoid needing by not training. Footnote 6 (p. 7) connects to the Dietterich/McNemar comparison. |
| `Cross-ValidationWhatDoesItEstimateandHowWellDoesItDoIt .pdf` (Bates, Hastie & Tibshirani, JASA 2024) | Object lesson in assumption-checking: naive CV CIs undercover because fold errors are correlated (§1.1, §4.1) — i.e., what happens when independence silently fails. Also clarifies estimands (Err vs Err_XY): our deterministic-model framing in §0 is the corresponding "what exactly are we estimating" exercise. |
| `group assignment/LLM_evalaution_survey.pdf` (Chang et al.) | Metric landscape; documents accuracy/error-rate and refusal-rate ("proportion of cases where LLMs refuse to answer", p. 22) as standard LLM metrics — external precedent for our accuracy + E-rate choices. No formal tests — which is partly why a statistics-first evaluation is a contribution. |
| `group assignment/socio_technical_evalaution_AI.pdf` | Framing only (capability evaluations as one layer of a larger safety picture); no statistical methods used by us. |
| `individual assignment/02445_Assignment_Jun2026.pdf` | Task 2 = design a CV scheme for repeated-measures data — the proper home of the two CV papers and Raschka §3. Useful contrast to keep straight at the oral: that problem trains models (CV needed); ours evaluates a fixed model (binomial family suffices). |

---

## 11. Known limitations to state in the report (kept in sync with CLAUDE.md)

1. Paired set is 114 tables vs. 13 figures → the primary McNemar mostly measures a
   *table*-reading penalty; B-only is underpowered (power check quantifies it).
2. The unpaired text-vs-graph comparison confounds question difficulty/topic with
   modality — descriptive, not causal.
3. "Type B" in the between-type chi-square = the 13 text-faithful B items only
   (selection bias).
4. Wilson CIs beside the McNemar output are marginal (ignore pairing).
5. Public past exams may be in Gemma's training data (contamination; §8).
6. Within-exam topic clustering → independence approximate for the unpaired tests (§8).
