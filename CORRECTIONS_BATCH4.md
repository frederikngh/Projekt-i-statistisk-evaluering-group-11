# Corrections batch 4 (2026-06-10)

Outcome of a full multi-agent review of the dataset and pipeline (every question
re-verified against its exam + solution PDF, every crop viewed, every text
representation compared digit-by-digit, code audited). Unlike batches 1-3
(`apply_corrections.py`), batch 4 was applied directly to `data/encoded/*.json`
by per-exam fix agents, each independently re-verified against the PDFs; the
git diff is the authoritative record. `questions.csv` and all crops were
regenerated afterwards.

Answer keys were confirmed correct for all 405 questions (15 x 27/27 vs the
solution grids and worked solutions). No `correct_answer` changed in batch 4.

## A. Stems de-embedded (design-B repair, the big one)

The encoder had pasted the referenced table/figure DATA into `question_text`
for ~70 of the 127 paired McNemar items (and some screenshot-only items).
Since `question_text` is sent as text in EVERY modality, the screenshot arm
also received the data as text - nullifying the image-vs-text manipulation for
those pairs. All such stems were restored to the faithful printed exam stem +
options (verified word-by-word against the exam PDFs); the data now reaches the
model ONLY via the crop (screenshot arm) or `text_representation` (text_desc arm).

- Fall2017: Q4, Q8, Q9, Q12*, Q13, Q18, Q19, Q20, Q21; also Q1, Q27
  (screenshot-only stems restored). *Q12: only the encoder-added x8=6.9
  sentence; the GMM parameters are printed in the exam stem and stay.
- Fall2018: Q9, Q10, Q14, Q15, Q16, Q19, Q20, Q21.
- Fall2019: Q6, Q9, Q10, Q11, Q12, Q14, Q15, Q16, Q17, Q18, Q19, Q22.
- Fall2020: Q8, Q11, Q12, Q13, Q15, Q16, Q17, Q19, Q23; also Q20, Q24.
- Fall2023: Q8, Q9, Q11, Q12, Q15, Q17, Q18, Q19.
- Fall2024: Q4, Q11, Q18; also Q23 (added ground-truth sentence removed).
- Spring2017: Q6, Q10, Q14, Q18, Q19, Q21, Q22, Q24, Q25; also Q2, Q4, Q5,
  Q7, Q11, Q12, Q26 (added figure-summary sentences removed; the deliberate
  batch-2/3 V-matrix inline in Q4 stays).
- Spring2018: Q14; Q26 (encoder-added tree description - which inverted node C -
  removed entirely).
- Spring2020: Q6, Q8, Q10, Q12, Q16, Q17, Q18, Q19, Q20, Q21, Q24, Q27; also
  Q11 ("(with AUC = 0.958)" removed).
- Spring2021: Q13, Q14 (encoder-added split-rule parentheticals removed).
- Spring2024: Q6, Q13, Q15, Q16, Q21, Q24; also Q1, Q7, Q22, Q25, Q27
  (added figure descriptions / solution coordinates removed).

## B. Unanswerable / wrong-content questions repaired

- Fall2018 Q3: V matrix (Equation (1), printed only in Q2) inlined into the stem.
- Fall2019 Q4: same V-matrix inline from Q2.
- Fall2019 Q5: attribute mapping added (x1 = MONTH, x2 = PM2.5; per Table 1).
- Fall2019 Q10/Q11/Q12: Table 5 row o4 corrected to 0 1 0 1 0 0 0 1 0 1 0
  (was wrong at f4/f5/f8/f9; under the old row Q11's keyed answer was underivable).
- Fall2019 Q15/Q16/Q20/Q21/Q25: figure references renumbered from solution-PDF
  numbering to exam numbering (tree = Fig 7, ANN outputs = Fig 8, CV curves =
  Fig 9, GMM scatter = Fig 10); source_location updated to match; crops now
  contain the right figures (Q20/Q21/Q25 previously delivered the WRONG figure).
- Fall2021 Q14: exam's truncated sentence completed so N = 11 / M = 8 reach both arms.
- Fall2021 Q17: "(The dataset has nine classes, y = 1, ..., 9.)" added (needed; was
  only in Table 1's caption).
- Fall2023 Q6: options A/B restored to b2^3 (was b2^2; key unaffected).
- Fall2017 Q18: option D's missing itemset {HL,WL,FG<=45%} restored (15 itemsets
  as printed; key unaffected). Found by a batch-4 verifier, beyond the original review.
- Fall2018 Q15: option C's two missing itemsets {f2,f3,f9}, {f6,f7,f9} restored.
- Spring2018 Q26: stem's inverted tree description removed (see section A).
- Spring2019 Q3: attribute-name mapping added (x5 = museums ... x9 = religious).
- Spring2019 Q9: per-class totals (263/359/358, N = 980) inlined into the stem
  (both arms); pre-computed complements removed from text_representation.
- Spring2020 Q27: Table 6 corrected - y1_test row (0,1,0,0) (the keyed answer D
  depends on it) and training row y5 (0,0,1,1).
- Spring2023 Q6: text_representation now contains the full 10x10 distance matrix
  (copied verbatim from Q5) instead of a cross-reference the model never saw.
- Spring2023 Q15: distractor norm subscripts corrected (option C node B -> L1;
  option D node A -> L2, node B -> L1).
- Spring2023 Q25: re-typed A -> B screenshot-only; the dropped exam sentence
  referencing Figure 10 restored; new crop Spring2023_Q25.png. (As pure text the
  question was systematically answered wrong - the components' positions matter.)
- Spring2025 Q19: the exam's omitted Hint block (McNemar formulas) transcribed
  into the stem.

## C. text_representation leak trims (text arm had pre-computed help)

Fall2017 Q9 (3-NN set + distances), Q13 (TP/FN/FP/TN + class totals);
Fall2019 Q16 (first-accepted-split note), Q18 + Q19 (Total N = 981);
Spring2017 Q25 (TP/FN/FP/TN mapping); Spring2018 Q14 (mapping + totals);
Spring2019 Q8 (N = 980), Q9 (complement counts), Q22 (variable summary beyond
figure content), Q24 (initialization + misclassified set); Spring2021 Q14
(observation-to-node assignment); Fall2024 Q18 (60-points-per-fold note).
Spring2024 Q24 gained a Classes line (parity with its crop's caption after the
stem de-embed).

## D. Code fixes

- collect.py: extract_letter fallback now only accepts a line that IS the
  letter (a truncated reply can no longer score as "A" via the article "a");
  zero-byte results.csv gets a header; pre-flight check that every screenshot
  row's image exists.
- run_all.py: forwarded CSV path quoted (paths with spaces); a failing script
  now stops the run with exit 1.
- helpers.py: an explicitly given results path that does not exist is a fatal
  error (no silent fall-back to example data).
- power_check.py: low-discordant-count case now prints a message instead of nothing.
- simple_gemma.py: generation params via generate_kwargs (greedy), matching collect.py.
- crop_figures.py: column edges derive from each page's width (the hardcoded
  591pt right edge cut wide tables on US-Letter pages - Spring2024 Q16);
  clamp intersects the correct page's rect; +9pt headroom above figures
  (panel titles like "Option A" / "Histogram 1" are no longer sliced);
  EXTRA_REFS forces needed-but-unnamed regions into crops
  (Spring2017 Q13 + Figure 6; Spring2018 Q22 + Table 4; Spring2019 Q7 + Figure 4).

## E. Regenerated artifacts

- data/questions.csv: 532 rows = 134 text + 271 screenshot + 127 text_desc
  (Spring2023 Q25 moved text -> screenshot). validate_encoded.py: ALL OK.
- All 271 crops regenerated; ~160 changed (mostly the +9pt headroom; plus the
  Fall2019 renumbered figures, the three EXTRA_REFS crops, Spring2024 Q16's
  full table, the un-clipped panel titles, and the new Spring2023_Q25.png).
- data/results_noCoT_archive.csv: first partial run (no-CoT prompts; NOTE:
  generation params were silently ignored by the pipeline back then, so it
  SAMPLED at temperature 1.0 - not greedy).
- data/results_preFix_archive.csv: second partial run (CoT, 179 rows), collected
  against the pre-batch-4 dataset; superseded by the post-fix collection.

## F. Known accepted caveats (deliberately not changed)

- source_location page numbers follow the solution PDFs in several exams
  (metadata only; Fall2019's were fixed because the cropper can read them).
- Cosmetic left-edge column slivers on ~20 crops; Fall2021 Q27's crop shows the
  ROC figure twice (duplication only, nothing missing).
- Fall2017 Q1 keeps one encoder-added sentence naming the attributes (Table 1
  content; screenshot-only item, axes interpretation needs it).
- Spring2021 Q25 / Spring2024 Q10 stay text_faithful=false although their 1-D
  number lines are arguably transcribable (would have added 2 McNemar pairs).
- Exam-original quirks are preserved (e.g. Fall2021 Q15's denominator typo,
  Spring2017 Q3 / Spring2020 Q4 mentioning a figure that type-A delivery omits -
  both verified answerable without it; several solution-PDF prose typos where
  the answer grid is authoritative).
