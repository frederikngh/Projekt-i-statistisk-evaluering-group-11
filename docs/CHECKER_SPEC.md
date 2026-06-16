# Checking specification — independently verify an encoded DTU 02450 exam JSON

You are INDEPENDENTLY verifying ONE already-encoded exam JSON against its source PDFs.
This is ground-truth data defended in an oral exam. Be skeptical: RE-DERIVE facts from
the PDFs — do not just trust the JSON. Report discrepancies precisely; do not edit files.

Your per-exam inputs (exam label, JSON path, exam PDF, solution PDF) are in the message
pointing you here. Read PDFs with the Read tool (pass `pages`, max 20/request).

JSON schema (reference): each question has question_id, number, topic, type
(A = pure text / B = needs figure / C = needs table), text_faithful (bool), question_text
(stem + options A-D), options, correct_answer (A-D), text_representation (table
transcription, figure description, or null), needs_screenshot, screenshot_path,
source_location.

## Checks (priority order)
1. **ANSWER KEY (critical).** Independently extract the correct answer for ALL questions
   from the solution PDF: the front answer grid if present, AND the per-question worked
   solutions ("Solution N. The correct option is X."). Compare to each `correct_answer`
   and to `answer_key`. Report EVERY mismatch as `Qn: JSON=X, grid=Y, worked=Z`.
2. **CLASSIFICATION.** Check `type` and `text_faithful` per the rules: A = self-contained
   text/math; B = needs a figure OR the options are plots; C = needs a table/matrix.
   text_faithful = true only for tables / numeric-discrete figures (values printed);
   false for geometric figures (dendrogram, scatter, unlabeled histogram/bars, ROC,
   density/contour, decision surface) and whenever options are plots. Flag disagreements
   with a one-line reason; mark borderline ones "minor".
3. **TRANSCRIPTION (type C and faithful B).** Compare `text_representation` to the actual
   table/figure. Verify numbers: check every row of small tables; for large matrices check
   dimensions + a sample of cells + any cell the answer depends on. Report errors as
   `Qn: cell (i,j) JSON=.. PDF=..`.
4. **QUESTION TEXT.** Spot-check `question_text` stem + the four options against the exam
   PDF (meaning-preserving; ASCII math is fine). Report material differences (changed
   numbers, wrong/missing option, altered meaning); ignore cosmetic formatting.
5. **FIELD CONSISTENCY.** needs_screenshot/screenshot_path set for B/C and null for A;
   text_representation present iff text_faithful and type in {B,C}; options has A-D;
   correct_answer in A-D.

## Return (concise — do NOT paste the JSON, do NOT edit it)
- VERDICT: `PASS` (no issues) or `ISSUES (n)`.
- A bullet list of every discrepancy, each as `Qn [check]: detail`. Put ANSWER-KEY errors first.
- Note any classification you'd call differently, marked "minor" if borderline.
