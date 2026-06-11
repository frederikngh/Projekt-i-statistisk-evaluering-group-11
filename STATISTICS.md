# Statistics in this project — tests, theory, assumptions, and where the materials use them

Reference for writing the report's methods/results sections and for the individual oral.
The project description (step 5b) grades exactly this: *"Sufficiently describe why the
statistical test is relevant, what the assumptions for the test are, and if they are
completely/approximately met."* Every test section below answers those three questions,
and points to where the same method appears in the course materials (`materials/` in the
parent folder; page numbers are the printed page numbers).

**How to read this file:** each section explains the idea in plain words first, then gives
the formal machinery (same content, two altitudes). §0 explains the general logic that
every test shares — read it once and the rest follows. §13 at the bottom is the whole
analysis compressed to one page.

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

## 0. The ideas every test shares (plain words first)

**The basic problem.** We want to know Gemma's *true* accuracy — the fraction it would get
right if we could ask it infinitely many questions of this kind. We can't; we have a few
hundred. Any number we compute from a sample wobbles: a different set of 15 exams would
give a slightly different accuracy. Statistics is the toolbox for saying how much wobble
there is, and for deciding when a difference is too big to be explained by wobble alone.

**The hypothesis-testing recipe.** All five tests follow the same four steps (Raschka
lists them on p. 34):

1. **State the null hypothesis H0** — the boring explanation: "Gemma is just guessing",
   "the modality makes no difference", "all question types are equally hard". H1 is the
   interesting alternative. H0 is what we try to *reject*; we never "prove" it.
2. **Fix the significance level α = 0.05 before looking at the data.** α is the
   false-alarm rate we accept: if H0 is actually true, a 5 % chance that we wrongly cry
   "effect!".
3. **Compute a test statistic** — one number that summarizes how far the data is from what
   H0 predicts, chosen so that *if H0 were true* we know exactly (or approximately) how
   that number behaves. That known behaviour is called the **null distribution**, and it
   is the heart of every test: it tells us which values are ordinary and which are
   surprising under the boring explanation.
4. **Compute the p-value** = the probability, *assuming H0 is true*, of seeing a result at
   least as extreme as ours. Small p means: "if nothing were going on, data like this
   would be rare" — so we reject H0 when p < α.

**What a p-value is NOT** (classic oral trap): it is not the probability that H0 is true,
and 1 − p is not the probability that our finding is real. It is a statement about the
data under an assumption, not about the hypothesis.

**The two ways a test can fail** (stats cheat sheet, p. 5):

- **Type I error** — H0 is true but we reject it (false alarm). Probability = α.
- **Type II error** — H0 is false but we fail to reject it (a miss). Probability = β.
- **Power = 1 − β** — the probability of catching a real effect of a given size. A smoke
  detector analogy: α is how often it shrieks at burnt toast, power is how reliably it
  goes off at an actual fire of a given size. §8 quantifies our detectors.

**Confidence intervals.** A 95 % CI is the companion to the point estimate: the range of
true values that are *plausible* given the data. "95 %" describes the recipe, not one
interval: if we redid the whole experiment many times and built the interval the same way
each time, about 95 % of those intervals would contain the true accuracy. We report a CI
next to every accuracy because a test only says *whether* there is a difference; the CI
says *how big* and *how uncertain* (step 5b asks for exactly this "stability").

---

## 1. The probability model under everything

**Plain words.** Every row in `results.csv` is one prompt with a binary outcome: correct
(1) or wrong (0). Think of each question as one flip of a biased coin whose
heads-probability p is Gemma's true accuracy. Flip n independent coins and the number of
heads follows the **binomial distribution** — that single model is the engine inside the
binomial test, the Wilson interval, the z-test, the chi-squares and McNemar alike. They
differ only in *which groups of coin flips* they compare.

**Formally** (Raschka §1.7, pp. 10–11; same distribution on the stats cheat sheet, p. 6):
each answered question is a **Bernoulli trial**, so the number of correct answers out of
n is

$$X \sim \text{Binomial}(n, p), \qquad P(X = k) = \binom{n}{k} p^k (1-p)^{n-k},$$

where $\binom{n}{k}$ counts the ways to choose which k of the n questions are the correct
ones, $p^k$ is the probability of those k successes, and $(1-p)^{n-k}$ the probability of
the remaining failures.

Two things about *our* setup are worth saying precisely at the oral:

- **Where the randomness lives.** `collect.py` decodes greedily (`do_sample=False`), so
  Gemma's answer to a fixed prompt is deterministic and reproducible — re-asking the same
  question is not a new coin flip, it is the same flip replayed. The random quantity is
  therefore *which questions we have*: we treat the 405 exam questions as a sample from
  the (hypothetical) population of 02450-style MCQs, and p is Gemma's accuracy over that
  population. This is the standard framing for evaluating a fixed model on a test set
  (Raschka §1.3–1.7). We use *all* public exams 2017–2025, so strictly the inference
  target is the superpopulation of similar questions, not some larger pool we subsampled —
  say this honestly in the report.
- **Independence between trials.** One coin flip must not influence the next. Each prompt
  is a separate, stateless model call with no chat history, and the model weights are
  frozen — so no answer can causally affect another. This is exactly the precaution the
  exemplar paper took ("For every prompt a new chat was opened to ensure no dependence on
  previous prompts", LLM_bias_1_revision.pdf, App. C.1, p. 17) and what the project
  description's step 3 worries about for online tools (which learn from your prompts — a
  local model removes that problem entirely). Residual caveat: questions from the same
  exam share topics, so within-exam correlation can't be fully excluded; see §9.

**Why chance = 25 %.** Every question has exactly four scoreable options A–D (E is never
the keyed answer), so blind uniform guessing among A–D succeeds with probability ¼. If
Gemma instead spread its guesses over all five letters, its accuracy would be 20 % —
*lower* than 25 %. Testing against the higher value 25 % is therefore the **conservative**
choice: it can only make "better than guessing" harder to claim, never easier.

---

## 2. Accuracy with Wilson confidence intervals (`helpers.py`)

**Purpose.** Step 5b explicitly asks us to "present the stability (uncertainty/variance)
of the performance". A point accuracy of, say, 81.8 % (9/11) means little on its own —
with 11 questions, the truth could plausibly be anywhere from "barely above half" to
"nearly perfect". The CI makes that explicit; we attach one to every accuracy we print.

**Plain words.** The textbook interval just takes accuracy ± about two standard errors.
That works fine in the middle of the scale with lots of data, but it misbehaves exactly
where we may live: small groups (11 text items in the example data, 13 paired type-B
questions) and accuracies near 0 % or 100 %. The Wilson interval is a better-engineered
version of the same idea: it is pulled slightly toward 50 % (where uncertainty is
largest), it is asymmetric when the data demands it, and it can never leave the 0–100 %
range.

**Concrete demonstration** (the example from the `acc_text` docstring): 9 of 11 correct.
- Wald: $0.818 \pm 1.96\sqrt{0.818 \cdot 0.182/11} = 0.818 \pm 0.228$ → **[59.0 %, 104.6 %]** —
  an interval that includes accuracies above 100 %, which is nonsense.
- Wilson: **[52.3 %, 94.9 %]** — asymmetric (29.5 points down, 13.1 up, because from 81.8 %
  there is more room to be wrong downward than upward) and inside [0, 1].

**Theory.** The normal-approximation (Wald) interval that Raschka derives in §1.7
(eq. 10–17, pp. 10–11) is $\hat p \pm z\sqrt{\hat p(1-\hat p)/n}$ with z = 1.96, where
$\sqrt{\hat p(1-\hat p)/n}$ is the **standard error** — the typical sample-to-sample
wobble of an accuracy estimated from n items (cheat sheet, p. 1). The **Wilson (1927)
interval** instead inverts the score test (it asks: *which true values p would NOT be
rejected by the data?*), which gives centre and half-width

$$\frac{\hat p + z^2/2n}{1 + z^2/n} \;\pm\; \frac{z}{1 + z^2/n}\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}.$$

For large n the extra $z^2/n$ terms vanish and Wilson ≈ Wald; for small n they supply the
pull toward ½ and the boundary-respecting behaviour seen above.

**Assumptions.** Independent Bernoulli trials with common p (the §1 model). No normality
assumption on the raw 0/1 data — the approximation concerns the *proportion*, and Wilson
is precisely the variant that stays accurate at small n.

**Where the materials use it.** The cross-validation paper by Bayle et al. (NeurIPS 2020,
`Cross_Validation_Confidence_Intervals_For_Test_Error.pdf`, §5, p. 7) uses exactly this:
"95 % Wilson intervals, which are known to provide more accurate coverage for binomial
proportions than a ±2 standard error interval", citing Wilson (1927) and Brown, Cai &
DasGupta (2001). Raschka §1.7 gives the Wald interval that Wilson improves on.

**Implementation.** `statsmodels.stats.proportion.proportion_confint(k, n, method="wilson")`.

**Caveat to state.** The Wilson CIs printed next to the McNemar result are *marginal*
(they treat the image-arm and text-arm accuracies as separate samples and ignore the
pairing). Fine as descriptives; a CI on the paired *difference* would need a paired method
(e.g. bootstrap over pairs — see §10).

---

## 3. Test 1 — exact binomial test against chance (`binomial_test.py`)

**Purpose.** Manipulation check before anything comparative: per modality (text,
text_desc, screenshot), is Gemma doing better than random guessing? If a modality is at
chance, comparative results for it mean something very different.

**Plain words.** Suppose Gemma gets 9 of 11 text questions right. A guesser averages
0.25 × 11 ≈ 2.75 correct. Could a guesser get 9 just by luck? The binomial distribution
answers that exactly: add up the probability of every outcome at least as good
(9, 10 or 11 correct) for a Binomial(11, 0.25) coin. That tail sum is the p-value — here
about 0.0001, so luck is not a credible explanation and we reject "it's guessing". No
approximations are involved; we literally enumerate the lucky outcomes, which is why the
test is called *exact* and is trustworthy at any sample size.

**Hypotheses.** H0: p = 0.25. H1: p > 0.25, **one-sided** — decided a priori because
"above chance" is the only direction that changes our interpretation (see §9 on
sidedness).

**Theory.** Under H0 the correct-count is exactly Binomial(n, 0.25) — see §1; the p-value
is the exact upper tail sum

$$P(X \ge k_{obs}) = \sum_{i=k_{obs}}^{n}\binom{n}{i}\,0.25^i\,0.75^{\,n-i}.$$

A nice property: under *guessing*, every question has success probability exactly 0.25 no
matter how hard it is — difficulty only exists for a model that actually reads the
question. So heterogeneous question difficulty does not distort the null distribution at
all; it only matters under H1, where it affects power, not validity.

**Assumptions.**
1. Binary outcome per trial — holds by construction (answer matches key or not).
2. Fixed number of trials n — holds (the question set is fixed before collection).
3. Independent trials — argued in §1; the within-exam clustering caveat applies (§9).
4. Same null success probability 0.25 each trial — holds under H0 by the guessing argument
   above (and 25 % is conservative given the E option, §1).

**Where the materials use it.** The binomial distribution and its tail computation are on
the cheat sheet (p. 6). Raschka builds the same "correct predictions ~ binomial" model in
§1.7 (p. 10) and uses **exact binomial p-values** in §4.4 (pp. 37–38) — there as the exact
version of McNemar, i.e. literally the computation our test 2 reuses.

**Implementation.** `scipy.stats.binomtest(k, n, 0.25, alternative="greater")`.

---

## 4. Test 2 (PRIMARY) — McNemar's exact test, paired (`mcnemar_test.py`)

**Purpose.** The core of the project. Each type-B/C question with a faithful text version
was asked twice: figure/table as a cropped **image** vs. the same information written as
**text**, with the stem and options identical in both arms (design B). The *only* thing
that changes is the modality, so a paired comparison isolates the graph/table-reading
penalty from question difficulty.

**Plain words — why pairing, and why only disagreements count.** Comparing two groups of
*different* questions is always haunted by "maybe one group was just harder". Pairing
kills that objection: every question is compared *with itself*, once as image, once as
text — each question is its own control. Now sort the pairs into four buckets: both arms
right, both wrong, only-image right, only-text right. A question that both arms get right
(too easy) or both get wrong (too hard) tells us nothing about which *modality* is
stronger — it cancels out. All the evidence sits in the **discordant pairs**, where the
arms disagree. And here is the trick that makes the test simple: *if modality truly didn't
matter* (H0), then each disagreement is equally likely to fall either way — only-image and
only-text are 50/50, like flipping a fair coin. So the question "does modality matter?"
becomes "is this coin fair?", which is just a binomial test with p = ½.

**Worked example** (the synthetic `data/example.csv` output): 26 pairs split into
both = 10, only-image = 10, only-text = 6, neither = 0. The 10 + 0 = 10 concordant pairs
drop out; the coin was flipped n_d = 16 times and came up "image" 10 times. Ten heads in
16 fair flips is unremarkable — the exact two-sided p-value is 0.454 — so the example data
gives no evidence that modality matters. (For contrast: 13 of 16 the same way would give
p ≈ 0.021 — that is where significance starts, as `power_check.py` reports.)

**Hypotheses.** H0: P(correct as image) = P(correct as text) for the paired questions.
H1: the probabilities differ (two-sided).

**Theory.** Cross-tabulate each pair (Raschka Fig. 19, p. 36):

|                | text correct | text wrong |
|----------------|--------------|------------|
| **image correct** | a (both)   | b (only image) |
| **image wrong**   | c (only text) | d (neither) |

Under H0, each discordant pair favours image with probability ½, so with n_d = b + c

$$b \mid n_d \sim \text{Binomial}(n_d, \tfrac12),$$

and the exact two-sided p-value is $p = 2\sum_{i=b}^{n_d}\binom{n_d}{i}0.5^{n_d}$ (for
b > c; Raschka §4.4, p. 38 — note $0.5^i \cdot 0.5^{n_d-i} = 0.5^{n_d}$). This
conditioning-on-disagreement is precisely what removes per-question difficulty as a
confounder — the strongest sentence to say at the oral.

The large-sample version is McNemar's chi-square (Raschka §4.3, p. 37):
$\chi^2 = (b-c)^2/(b+c) \sim \chi^2_1$. It is the same coin-fairness check in disguise:
standardize the binomial count, $(b - n_d/2)\big/\sqrt{n_d/4}$ (observed heads minus
expected, over the standard deviation of a fair-coin count), and square it — a squared
standard normal is by definition χ² with 1 degree of freedom. Edwards' continuity-corrected
variant is $(|b-c|-1)^2/(b+c)$; the correction exists because a discrete staircase
(binomial counts) is being approximated by a smooth curve (χ²), and shaving half a step
off the difference compensates. Raschka's rule of thumb: the χ² approximation is
reasonable when b + c > 25, and the corrected version matches the exact test well when
b, c > 50. Our discordant counts may be small, **which is why we run the exact version** —
`mcnemar(table, exact=True)` — and never have to defend an approximation.

**Assumptions.**
1. **Pairs are independent of each other** — each pair is one exam question, answered in
   two stateless calls (§1). One question's outcome cannot influence another's.
2. **Within a pair, the two arms answer the same question** — guaranteed by design B: the
   stem and options are byte-identical, only figure-as-image vs. figure-as-text differs.
   (This was audited: an early encoding bug had table data embedded in ~70 stems, sending
   it to BOTH arms and nullifying the contrast; fixed in corrections batch 4.)
3. The two arms of a pair MAY be dependent — that is the point of pairing, not a
   violation. (Easy questions make both arms likely-right together; pairing exploits this
   shared component instead of being broken by it.)
4. No order/carry-over effects — holds because the arms are separate calls with no shared
   state (a human answering twice would have a memory confound; the model does not).

**Multiple testing.** We run McNemar three times (B only, C only, B+C — nested subsets).
Three chances at α = 0.05 means a higher family-wise chance of at least one false alarm
(for three independent tests it would be 1 − 0.95³ ≈ 14 %). Our handling: **B+C is
declared the primary confirmatory test up front; the per-type runs are exploratory.** This
pre-declaration is the clean alternative to a Bonferroni correction (α/m) and follows
Raschka §4.5 (pp. 38–39): either control the family-wise error or clearly separate
confirmatory from exploratory analyses.

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

## 5. Test 3 — two-proportion z-test, unpaired (`text_vs_graph_test.py`)

**Purpose.** Geometric figures (scatter plots, dendrograms, ROC curves, contours…) have no
faithful text form, so they cannot enter the paired design. We compare type-A pure-text
questions (n = 134) against these screenshot-only type-B questions (n = 144) as two
independent groups.

**Plain words.** Two separate accuracies, one gap between them. The question is whether
the gap is bigger than the wobble two samples of this size would show even if both groups
had the same true accuracy. The yardstick for that wobble is the **standard error of the
difference**; the test statistic z simply counts *how many standard errors wide the
observed gap is*. Under H0, z behaves like a standard normal, so |z| > 1.96 happens only
5 % of the time by chance — that is the rejection rule.

*Hypothetical worked example:* text 70 % of 134 vs. graph 55 % of 144. Pooled accuracy =
(94 + 79)/278 ≈ 0.62; SE = $\sqrt{0.62 \cdot 0.38 \,(1/134 + 1/144)}$ ≈ 0.058; z ≈
0.15/0.058 ≈ 2.6 standard errors → p ≈ 0.01: a 15-point gap with these group sizes would
be hard to blame on sampling wobble.

**Hypotheses.** H0: p_text = p_graph. H1: they differ (two-sided).

**Theory.** With independent groups, $\hat p_1 - \hat p_2$ is approximately normal for
moderate n — this is the **central limit theorem** doing the work: a sum of many
independent 0/1 outcomes is approximately bell-shaped, hence so is each $\hat p$ and their
difference. Under H0 both groups share one p, estimated by pooling
$\hat p = (k_1+k_2)/(n_1+n_2)$ (use *all* the data to estimate the one accuracy H0 says
exists), giving

$$z = \frac{\hat p_1 - \hat p_2}{\sqrt{\hat p(1-\hat p)\left(\frac{1}{n_1}+\frac{1}{n_2}\right)}} \;\sim\; N(0,1) \text{ under } H_0.$$

On a 2×2 table this z-test and the chi-square test are the same test: **z² = χ²** (the
uncorrected χ²) — our defence line for using a z-test when the project description's
step 5b lists "Chi-square test (for proportions)". It is also the proportions analogue of
the cheat sheet's **unpaired two-sample t-test** (pp. 2–3): same logic of
difference-over-pooled-standard-error, with the binomial supplying the variance (for 0/1
data the variance p(1−p) is determined by the mean, so no separate variance estimate is
needed).

**Assumptions.**
1. Independent observations within and *between* groups — holds: the two groups are
   disjoint question sets, every item answered once. (Note: Raschka §4.2 (pp. 34–35) and
   Dietterich criticize this test heavily — but for the scenario of *two models on the
   same test set*, where the two proportions are computed on identical items and are
   therefore dependent, inflating false positives. That specific violation does not apply
   to us: one model, two disjoint item sets. Worth pre-empting at the oral.)
2. Normal approximation adequacy — the bell-curve approximation needs enough expected
   successes AND failures in each group; rule of thumb ≥ ~5–10 of each. With n = 134 and
   144 this holds unless accuracy is extreme (check against the observed counts when the
   data is in).
3. Common p within each group under H0 — same superpopulation framing as §1.

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

## 6. Test 4 — chi-square test of homogeneity across question types (`question_types_test.py`)

**Purpose.** With modality held fixed at *text* (type A as plain text, B/C as their text
descriptions), does accuracy depend on question *type*? This separates "the content type
is hard" from "the image rendering is hard".

**Plain words.** Lay out a table: one row per type (A, B, C), columns "correct" and
"wrong". If type made no difference, every row should be right at the *same* rate — the
overall rate. From that we can fill in the table we would **expect** under H0: each cell =
its row total × the overall correct (or wrong) share. The χ² statistic then walks through
the cells and adds up the squared mismatches between observed and expected, each scaled by
the expected count: $(O-E)^2/E$. Scaling by E matters — being off by 5 answers is shocking
in a cell expecting 4, trivial in a cell expecting 400. Big total mismatch ⇒ the
"one common accuracy" story doesn't fit the table ⇒ small p-value.

**Hypotheses.** H0: p_A = p_B = p_C. H1: at least one differs.

**Theory.** Build the 3×2 table of (correct, wrong) counts per type. Expected counts under
independence: $E_{ij} = (\text{row}_i \times \text{col}_j)/\text{total}$, statistic
$\chi^2 = \sum (O-E)^2/E$ with df = (r−1)(c−1) = 2. This is the cheat sheet's chi-square
test of independence verbatim (pp. 4–5, including the expected-count formula and the
fitness/smoking worked example). Why the χ² distribution: each standardized cell deviation
is approximately standard normal, and a sum of squared standard normals is χ² by
definition. Why df = 2 and not 6: the margins (row and column totals) are fixed when
computing E, so once two cells are known the rest follow — only 2 cells are free to vary,
hence 2 degrees of freedom.

**Assumptions.**
1. Independent observations, each in exactly one cell — holds (each row of the table is
   one question answered once; types are mutually exclusive).
2. **All expected counts ≥ 5** — the classical validity rule (cheat sheet p. 4). It exists
   because χ² is a smooth approximation to discrete counts, and the approximation needs
   enough data per cell to hold. The script computes `expected.min()` and prints a warning
   when violated. With only 13 type-B items this is the cell to watch; if it fails, report
   the warning and interpret descriptively (or merge B into B+C).
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

## 7. Test 5 — E-rate ("don't know") chi-square, 2×2 (`dont_know_test.py`)

**Purpose.** Every prompt offers E = "don't know". If Gemma has any awareness of when it
cannot read a figure, E should be more frequent on image input than on text input. This is
our taught-statistics substitute for confidence calibration (which we descoped — §10): it
measures *expressed* uncertainty behaviourally instead of reading internal probabilities.

**Plain words.** Same machinery as test 4, smaller table: rows = input kind (image vs.
text), columns = (said E, said A–D). H0 says the E-rate is one number regardless of input;
the test checks whether the observed 2×2 table is too lopsided for that story. Note the
outcome being modelled has changed: the coin flip here is "said E or not", not "correct or
not" — a different Bernoulli variable on the same calls.

**Hypotheses.** H0: P(answer = E | image input) = P(answer = E | text input). Two-sided.

**Theory.** χ² as in test 6 with df = (2−1)(2−1) = 1.

**Assumptions.** As in test 4: independent observations, expected counts ≥ 5. The
expected-count rule matters *more* here because E may be rare (in the no-CoT pilot Gemma
said E mostly on broken questions; possibly never after the fixes). The script:
- prints "Gemma never answered E — nothing to test (a finding in itself)" when there are
  zero E's overall: a model that *never* admits uncertainty on unreadable figures is a
  reportable behavioural result, no p-value needed;
- prints the expected-count warning otherwise. If expected counts are < 5, the standard
  remedy to name at the oral is **Fisher's exact test** (`scipy.stats.fisher_exact`),
  which — like the exact binomial — enumerates tables directly (conditioning on the
  margins) instead of using the χ² approximation, and is valid at any counts.

**Implementation detail worth knowing.** `scipy.stats.chi2_contingency` applies **Yates'
continuity correction by default on 2×2 tables** (df = 1), so the printed χ² here is the
corrected one — slightly conservative, appropriate at small counts. (Same
discrete-staircase-vs-smooth-curve fix as Edwards' correction for McNemar, Raschka p. 37.)

**Where the materials use it.** Chi-square anchors as in test 4 (cheat sheet pp. 4–5,
slides part 1 slide 18). The LLM-evaluation survey describes refusal-rate as an
established metric: "the metric … is the proportion of cases where LLMs refuse to answer"
(LLM_evalaution_survey.pdf, p. 22, on the TrustGPT PVA evaluation) — our E-rate is that
metric with an explicit opt-out option.

---

## 8. Power check — minimum detectable effects (`power_check.py`)

**Purpose.** Not a hypothesis test: a *design* justification. Step 4d of the project
description asks "How are you deciding on the number of times you will prompt the GenAI?
(Hint: sample size estimation, if applicable.)" Our item budget is fixed by reality (15
public exams × 27 questions, of which 127 are pairable and 134/144 form the unpaired
groups), so instead of choosing n for a target effect, we answer the inverse question:
**given our n, how big must the true effect be before our tests would reliably detect
it?** Both directions are standard power analysis.

**Plain words.** A non-significant result has two possible readings: "there is no effect"
or "there is an effect but our sample was too small to see it". Power analysis is what
separates them. Power = the probability our test rejects H0 *given* a true effect of a
stated size (the smoke detector's sensitivity to a fire of a stated size, §0). By
convention we want ≥ 80 %. Reporting the **minimum detectable effect** — the smallest gap
our setup catches with 80 % power — tells the reader exactly which effects our study could
and could not have found, which turns a null result from a shrug into a statement.

**Concepts** (stats cheat sheet, p. 5): Type I error = rejecting a true H0, probability
α = 0.05; Type II error = failing to reject a false H0, probability β; **power = 1 − β**,
with 80 % the usual convention.

**Part 1 — McNemar.** The exact McNemar test only uses the n_d discordant pairs (§4), so
its sensitivity is fully described by one number: out of n_d disagreements, how many must
favour the same modality before the exact binomial p drops below 0.05? The script scans
k = ⌈n_d/2⌉+1 … n_d and reports the first k that rejects. On the example data: 16
disagreements → at least 13 must point the same way (13/16 gives p ≈ 0.021; 12/16 only
p ≈ 0.077). If n_d is tiny, the script instead reports that significance is impossible —
the honest statement that the B-only analysis is underpowered.

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

## 9. Cross-cutting assumptions and choices (the oral lives here)

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

**One-sided vs. two-sided.** A one-sided test spends all of α on one direction: it is more
sensitive there, blind the other way, and only legitimate if the direction was fixed
before seeing data. Binomial test: one-sided (H1: better than chance), fixed a priori —
below-chance performance has no separate scientific story here, and the directional
question is the only one we'd act on. McNemar and z-test: two-sided — we *expect* images
to be worse, but the tests stay agnostic; a two-sided test is conservative and avoids any
suspicion of choosing the direction after seeing data. (The cheat sheet p. 2 explains
exactly this halving/doubling of tail probability.)

**Multiple testing across the analysis.** Every test run at α = 0.05 is another 5 % lottery
ticket for a false alarm; run enough tests and "something" will be significant by chance
(three independent tests already push the family-wise false-alarm rate toward
1 − 0.95³ ≈ 14 %). Our handling: one pre-declared primary confirmatory result — **McNemar
on B+C**. Everything else (per-type McNemars, the unpaired z, the type chi-square, the
E-rate) is supporting/exploratory and is reported with that label rather than with a
correction. If an examiner wants a correction anyway: Bonferroni α/m (Raschka §4.5,
pp. 38–41, including Perneger's caveat that Bonferroni buys Type-I control at the cost of
Type-II errors).

**p-values AND intervals.** Every accuracy is reported with its Wilson CI, every test
with its p-value (formatted by `fmt_p`: "p < 0.001" or rounded to 3 decimals, plus the
α = 0.05 verdict). The CI answers "how big and how uncertain", the p-value answers "is it
distinguishable from H0" — step 5b asks for both.

**Training-data contamination.** The exams are public PDFs from 2017–2025; Gemma may have
seen them in pre-training. This would inflate *absolute* accuracy (test 1) and is named in
the report. The paired *contrast* (test 2) is partially protected — both arms share the
identical stem, so memorization helps both arms; only the figure rendition differs — but
"partially" is the right word, since a memorized figure discussion could leak
asymmetrically.

**Internal validity constants.** One model (`gemma-4-E2B-it`) for all 532 calls, one
decoding setting (greedy, same token budget), one prompt template per modality. Any
between-arm difference is therefore attributable to the input rendition, not to runtime
configuration drift.

---

## 10. Methods in the materials we deliberately do NOT use — and why

Likely oral questions ("you read about X — why isn't it in your analysis?"):

- **Pearson / Spearman correlation** (slides part 1, slides 6–17, with the P/A
  purpose-assumption format). Measures association between two *paired numeric* variables.
  We have a binary outcome vs. categorical factors — nothing to correlate. (If we had kept
  model confidence, correct-vs-confidence would have been its use case; descoped.)
- **t-tests / ANOVA** (cheat sheet pp. 2–3; slides slide 18; exemplar §3.2). Right tool
  for continuous/count outcomes; wrong distribution family for 0/1 — see §9. The
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
  of §1 applies directly. Raschka's own summary points here (p. 43): McNemar is the
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

## 11. Map of the materials

| Material | What it contributes to our stats |
|----------|----------------------------------|
| `group assignment/02445_Project_June2026.pdf` | The grading contract: step 5b (justify test, state assumptions, check them; "present stability/uncertainty"), step 4d (sample-size estimation → our power check), step 3 (prompt independence). Lists t-test / chi-square / ANOVA as the expected level of tooling. |
| `group assignment/LLM_bias_1_revision.pdf` (exemplar, Due et al. FAccT '24) | The rigor template: model written out (two-way ANOVA, §3.2 p. 8), assumptions checked (QQ-plots, Box-Cox), independence by fresh chat per prompt (App. C.1), pilot → Cohen's d → power-based n = 80/group, results table w/ F and p (Table 2, p. 10). Our binary-outcome translation of each element: §1, §8, and the per-test assumption blocks above. |
| `stats_cheat_sheet.pdf` | The 02402 backbone: t-tests incl. the paired/unpaired distinction that motivates McNemar-vs-z (pp. 2–3), chi-square GoF + independence with expected counts and df (pp. 3–5), Type I/II errors (p. 5), the binomial distribution (p. 6), t and χ² tables (pp. 7–8). |
| `other/02445_2026_part1.pdf` (course slides) | The course's P:/A: (purpose/assumptions) presentation style we mirror here; Pearson vs. Spearman (slides 6–17); the test refresher incl. "chi-square: is performance independent of the dataset used" (slide 18); the assumptions-not-met ladder (slide 19). |
| `raschka_StatEva_modelComparison.pdf` | The ML-evaluation theory source: accuracy as binomial + normal-approx CI (§1.7 pp. 10–11), the 4-step testing recipe and difference-of-proportions z-test with its limits (§4.2 pp. 34–35), McNemar χ², continuity correction, exact binomial form (§4.3–4.4 pp. 35–38), multiple testing/Bonferroni/omnibus-then-post-hoc (§4.5 pp. 38–41), Cochran's Q (§4.6), Dietterich's test comparison (p. 43), effect size vs. p-value (§4.13 p. 45). |
| `Cross_Validation_Confidence_Intervals_For_Test_Error.pdf` (Bayle et al., NeurIPS 2020) | Uses 95 % **Wilson intervals** for binomial proportions, citing Wilson 1927 and Brown et al. 2001 (§5 p. 7) — our CI choice, used for the same reason. Shows what valid inference takes when estimates ARE dependent (CV) — the machinery we avoid needing by not training. Footnote 6 (p. 7) connects to the Dietterich/McNemar comparison. |
| `Cross-ValidationWhatDoesItEstimateandHowWellDoesItDoIt .pdf` (Bates, Hastie & Tibshirani, JASA 2024) | Object lesson in assumption-checking: naive CV CIs undercover because fold errors are correlated (§1.1, §4.1) — i.e., what happens when independence silently fails. Also clarifies estimands (Err vs Err_XY): our deterministic-model framing in §1 is the corresponding "what exactly are we estimating" exercise. |
| `group assignment/LLM_evalaution_survey.pdf` (Chang et al.) | Metric landscape; documents accuracy/error-rate and refusal-rate ("proportion of cases where LLMs refuse to answer", p. 22) as standard LLM metrics — external precedent for our accuracy + E-rate choices. No formal tests — which is partly why a statistics-first evaluation is a contribution. |
| `group assignment/socio_technical_evalaution_AI.pdf` | Framing only (capability evaluations as one layer of a larger safety picture); no statistical methods used by us. |
| `individual assignment/02445_Assignment_Jun2026.pdf` | Task 2 = design a CV scheme for repeated-measures data — the proper home of the two CV papers and Raschka §3. Useful contrast to keep straight at the oral: that problem trains models (CV needed); ours evaluates a fixed model (binomial family suffices). |

---

## 12. Known limitations to state in the report (kept in sync with CLAUDE.md)

1. Paired set is 114 tables vs. 13 figures → the primary McNemar mostly measures a
   *table*-reading penalty; B-only is underpowered (power check quantifies it).
2. The unpaired text-vs-graph comparison confounds question difficulty/topic with
   modality — descriptive, not causal.
3. "Type B" in the between-type chi-square = the 13 text-faithful B items only
   (selection bias).
4. Wilson CIs beside the McNemar output are marginal (ignore pairing).
5. Public past exams may be in Gemma's training data (contamination; §9).
6. Within-exam topic clustering → independence approximate for the unpaired tests (§9).

---

## 13. Quick overview — the whole analysis on one page

**Setup.** One row per (question, modality); correct = Gemma's letter matches the key.
532 calls: 134 `text` (type A) + 271 `screenshot` + 127 `text_desc`; the 127 questions
with both image and text forms (114 tables, 13 figures) are the paired set; 134 pure-text
vs. 144 screenshot-only geometric figures are the unpaired groups. α = 0.05 everywhere;
every accuracy gets a 95 % Wilson CI; **primary confirmatory result = McNemar on B+C**,
all other tests are supporting/exploratory.

**The tests, one block each:**

1. **Better than guessing?** — exact binomial (`binomial_test.py`).
   H0: accuracy = 25 % (guessing over A–D); one-sided. Correct-count ~ Binomial(n, 0.25)
   under H0; p = exact upper tail. Exact at any n; 25 % is conservative because of E.
   Significant ⇒ the modality is above chance. `binomtest(k, n, 0.25, alternative="greater")`.

2. **Does modality matter on the SAME question?** — exact McNemar, **PRIMARY**
   (`mcnemar_test.py`). Pairs sorted into both-right / only-image / only-text /
   both-wrong; agreements carry no information; under H0 the n_d disagreements split
   50/50 like fair coin flips ⇒ b ~ Binomial(n_d, ½), exact two-sided p. Pairing makes
   each question its own control → difficulty cannot confound. Exact version because n_d
   may be < 25. Significant ⇒ the rendition (image vs. text) itself changes correctness.
   `mcnemar(table, exact=True)`.

3. **Pure text vs. pure graph questions** — two-proportion z (`text_vs_graph_test.py`).
   H0: equal accuracy; z = gap / pooled standard error ~ N(0,1); z² = (uncorrected) χ².
   Needs ≥ ~5–10 expected successes/failures per group (n = 134/144: fine). Different
   questions in the two groups ⇒ difficulty confounded ⇒ descriptive, not causal.
   `proportions_ztest([k1,k2],[n1,n2])`.

4. **Accuracy across types A/B/C (all text)** — chi-square 3×2 (`question_types_test.py`).
   H0: one common accuracy; χ² = Σ(O−E)²/E, E = row·col/total, df = 2. Omnibus only
   (says *that* types differ, not which); valid if all expected counts ≥ 5 — script
   warns if not (watch the 13-item type-B row). `chi2_contingency(table)`.

5. **"Don't know" more often on images?** — chi-square 2×2 on the E-rate
   (`dont_know_test.py`). H0: same E-rate for image and text input; df = 1,
   Yates-corrected by scipy default. If E is too rare (expected < 5): Fisher's exact is
   the named fallback; zero E's overall = a reportable finding by itself. Our
   within-the-curriculum substitute for calibration.

6. **What could we even detect?** — power check, not a test (`power_check.py`).
   McNemar: out of n_d disagreements, the smallest k pointing one way with p < 0.05
   (e.g. 13 of 16). z-test: smallest accuracy drop detectable with 80 % power at our
   group sizes. Answers project-description step 4d with the sample fixed by reality
   (15 exams × 27 questions).

**Mini-glossary.**
- **H0** — the no-effect explanation we try to reject.
- **p-value** — probability of data at least this extreme *if H0 is true* (not the
  probability that H0 is true).
- **α = 0.05** — the false-alarm rate we accept; reject H0 when p < α.
- **Power** — chance of detecting a real effect of a given size; we use the 80 %
  convention; Type I error = false alarm (α), Type II = miss (β), power = 1 − β.
- **95 % CI** — range of plausible true values; the recipe captures the truth in 95 % of
  repeated experiments. Wilson = the version that behaves at small n and near 0 %/100 %.
- **Exact test** — p-value computed by enumerating outcomes (binomial/Fisher), no
  large-sample approximation to defend.

**If you only remember three sentences:**
1. Every answer is a coin flip with unknown heads-probability — all our tests are the
   binomial model compared across different groups of flips.
2. The paired McNemar is the only comparison where question difficulty cannot confound,
   because every question is its own control and only disagreements count — that is why
   it is the primary test.
3. Significance is not size: the p-value says a difference is real, the Wilson CI says
   how big it plausibly is — the report always gives both.
