# Encoding specification — DTU 02450 ML exam → JSON

You are encoding ONE DTU "Introduction to Machine Learning" (course 02450) written
exam into a structured JSON file, for a statistical study of how well the Gemma
vision-language model answers MCQs in different modalities. ACCURACY IS CRITICAL —
this is ground-truth data defended in an oral exam. Work methodically; do not guess.

Your per-exam inputs (exam label, exam PDF path, solution PDF path, output JSON path,
page counts) are in the message that points you here. Read PDFs with the Read tool
(pass `pages`, max 20 pages/request).

## Format background
Each exam usually has 27 MCQs with options A, B, C, D and E = "Don't know." (E is never
correct; it's the model's abstain option). If an exam's count differs, encode every MCQ
present, in order, and note it. Classify each question by TYPE and whether it has a
faithful TEXT form:
- **type "A"** = pure text: self-contained text/math ("which statement is correct", or a
  computation whose inputs are printed inline as text/numbers/matrices). No figure/table.
- **type "B"** = needs a FIGURE/graph to answer (scatter, dendrogram, histogram, ROC,
  density/contour, decision surface, GMM samples), OR the answer options themselves are
  plots ("Dendrogram 1..4", "Plot 1..4").
- **type "C"** = needs a TABLE / numeric matrix to answer (distance matrix,
  binary/market-basket matrix, observations list, confusion matrix, KDE table, CV counts).

## Text-faithfulness (boolean `text_faithful`)
Can the ENTIRE question (stem + data + answer options) be written losslessly in plain text?
- type A: always true.
- type C: true — transcribe the numbers.
- type B: true ONLY if the figure is essentially NUMERIC/DISCRETE (e.g. a confusion matrix,
  or a histogram/bar chart whose bar values are PRINTED on the figure). For GEOMETRIC
  figures — dendrograms, scatter/point clouds, unlabeled histograms/bars, ROC curves,
  density/contour, decision surfaces, sample scatter — FALSE. If the answer OPTIONS are
  plots ("which dendrogram"), FALSE even if a numeric table is also present.
- **Rule of thumb:** numbers/discrete → true; geometry/shape → false. Classify by what is
  needed to ANSWER.

## Procedure
1. **Answer key.** In the solution PDF, near the front (page 1 or 2) there may be a grid
   titled "Answers:" mapping Q1..QN to letters. If present, record it. ALSO read the worked
   solutions (later pages), which each state "Solution N. The correct option is X."
   Cross-check the two. Some solution PDFs have NO grid — then use the worked-solution text.
   If grid and worked text disagree, trust the worked text and note it.
2. **Read the exam.** Page 1 is a bubble answer sheet; page 2 may be blank; questions start
   ~page 3 (footer "k of N"). For each question capture: full stem, options A-D (E always
   "Don't know."), a short topic, and which Table(s)/Figure(s) it uses.
3. **Classify**: type (A/B/C) + text_faithful, per above. Classify by what's needed to ANSWER.
4. **Text representation** (`text_representation`):
   - type C: transcribe the table/matrix as aligned plain text. ASCII only (mu, sigma, <=,
     ^, sum — NO LaTeX/unicode).
   - type B faithful: a faithful, NEUTRAL description of the numeric/discrete figure (don't
     bias toward the answer).
   - type B non-faithful, and type A: `null`.
5. **Write the output JSON** (single valid JSON object; double quotes; no trailing commas;
   no Python) with EXACTLY this schema (`<ExamLabel>` = your exam label):
```json
{
  "exam": "<ExamLabel>",
  "course": "02450",
  "source_pdf": "ML-examsets/<ExamLabel>.pdf",
  "solution_pdf": "ML-solutions/<ExamLabel>_sol.pdf",
  "n_questions": 27,
  "answer_key": {"Q1": "A", "...": "...", "Q27": "C"},
  "questions": [
    {
      "question_id": "Q1",
      "number": 1,
      "topic": "PCA",
      "type": "B",
      "text_faithful": false,
      "question_text": "<full stem>\n\nA. <opt A>\nB. <opt B>\nC. <opt C>\nD. <opt D>",
      "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
      "correct_answer": "B",
      "text_representation": null,
      "needs_screenshot": true,
      "screenshot_path": "data/screenshots/<ExamLabel>_Q1.png",
      "source_location": "exam p.1, Figure 1 (scatter-plot matrix)"
    }
  ]
}
```
Field rules:
- `needs_screenshot`: true for type B/C, false for type A.
- `screenshot_path`: `data/screenshots/<ExamLabel>_Q<N>.png` for type B/C; `null` for type A.
- `question_text`: faithful stem then the four options on separate lines (ASCII math) — this
  is what gets sent to the model.
- `options`: the four strings WITHOUT letter prefixes.
- `correct_answer`: a letter A-D (never E).
- `source_location`: page + Table/Figure number(s) — for the screenshot checklist.
- Encode ALL questions in order. If one is too complex to transcribe faithfully, still
  include it with your best effort and FLAG it in your summary.
6. **Validate**: run
   `python3 -c "import json;d=json.load(open('<OUT_JSON>'));assert len(d['questions'])==d['n_questions'];print('OK',len(d['questions']))"`

## Return (concise — DO NOT paste the JSON)
- the full answer key (e.g. "Q1=A Q2=B ...").
- counts: type A / B / C, and how many type-B are text_faithful vs not.
- screenshot checklist: each type B/C question_id + source_location.
- any items unsure how to classify (id + why); any not transcribable faithfully (id + reason).
- confirm the JSON validates with the expected number of questions.
