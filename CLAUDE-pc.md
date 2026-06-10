# 02445 Project — Gemma data collection (Windows PC session)

Context for Claude sessions on the user's Windows PC (RTX 5070 Ti). Read this first
when working on that machine. Its job: **run the 532-call Gemma collection and push
`data/results.csv`.** The full project context/history is in CLAUDE.md next to this
file; this one is the operational subset plus the rules that must not be broken.

## Commit rules

- **Commits are made as the user himself: `Frederik Naervig <frederikngh@gmail.com>`,
  with NO `Co-Authored-By` trailers.** If git identity is unset on this PC, set exactly
  that (`git config user.name` / `user.email`) before committing. Plain one-line
  commit messages.
- (History note: the repo's .md context files were local-only until 2026-06-10, when
  the user chose to publish them so this PC's session has full context. The
  no-trailer commit practice stays.)
- All tracked `.py` files are plain ASCII (no em-dashes/unicode). Keep it that way.

## What the project is (short)

DTU 02445 "Statistical Evaluation of AI", group 11, June 2026. Report due 24.06 12:00.
Research question: does Gemma 4 do worse on multiple-choice ML-exam questions when a
figure/table is given as an **image** vs. the same information **as text**? Outcome is
correct/wrong only (calibration was descoped). The stats (McNemar paired primary,
binomial vs chance, two-proportion z, chi-square, power) run on the Mac via `run_all.py`
— this PC only collects.

## The mission, in order

1. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
   **before** `pip install -r requirements.txt` (plain pip torch = CPU-only on Windows;
   the 5070 Ti is Blackwell/sm_120 and needs the cu128 build).
2. Hugging Face: the licence for `google/gemma-4-E4B-it` must be accepted on the model
   page (browser), then `hf auth login` once. Gated model — nothing downloads before this.
3. Smoke test: `python simple_gemma.py`. Must print `Using device: cuda`. First run
   downloads ~15 GB.
4. Full run: `python collect.py` (~532 calls, prints `[i/532]`). Resume-safe: rows
   already in `data/results.csv` are skipped, so Ctrl+C + rerun is fine.
   Flags if needed: `--exam Fall2024`, `--modality text|screenshot|text_desc`, `--dry-run`.
5. Sanity-check `data/results.csv` (see below), then commit it as the user with a plain
   message like "Add Gemma results from the full run" and push.

## Known risks and the agreed responses

- **CUDA out of memory** (E4B bf16 is tight in 16 GB VRAM): the agreed fallback is
  `google/gemma-4-E2B-it` for **ALL 532 calls** — one model consistently, never a mix.
  That means: delete any partial `data/results.csv` from E4B, change the model id in
  BOTH `collect.py` and `simple_gemma.py`, rerun everything. Tell the user it happened.
- **Thinking-mode contamination (the canary):** the `raw_response` column must be bare
  letters (A/B/C/D/E). Gemma 4 is a reasoner; if raw responses are sentences/reasoning,
  the parse is unreliable — STOP and investigate with the user before trusting any data.
- **Device says `cpu` or `mps` missing errors:** torch install went wrong; uninstall
  torch/torchvision and redo step 1. Do not run the collection on CPU.

## Hard rules for code changes

- **`build_prompt()` and `INSTRUCTION` in collect.py must stay byte-identical.** Any
  edit near them must be proven harmless:
  `python collect.py --dry-run > after.txt` and diff against a pre-edit dump.
  (The Mac sessions kept this invariant through every refactor; do not be the one
  who breaks it.)
- Greedy decoding stays (`do_sample=False`, `max_new_tokens=16`) — reproducibility.
- If you touch ANY project `.py`, match the deliberate 2nd-semester style: NO list
  comprehensions, no type hints, no pathlib/regex/Counter/defaultdict, plain loops +
  `append`, lists not tuples, string dict keys (`exam + " " + qid`), no chained calls
  on results (`result = binomtest(...)` then `result.pvalue`), no format specs —
  `round()` / `percent()` + string `+` concatenation, p-values via `fmt_p()`,
  copy-paste blocks fine, named matplotlib colors, plain ASCII.
  Exception: `crop_figures.py`, `build_questions_csv.py`, `build_screenshot_checklist.py`
  keep their more advanced style (one-time build tools).

## Repo facts

- `data/questions.csv` = the manifest collect.py reads: **532 rows = 135 `text` +
  270 `screenshot` + 127 `text_desc`** (15 exams x 27 questions, types A/B/C).
  Generated from `data/encoded/*.json` (the source of truth) — both are committed,
  so nothing needs regenerating on this PC.
- All 270 figure crops are committed in `data/screenshots/`; `questions.csv` uses
  relative paths, so **run everything from the repo root**.
- `data/results.csv` columns include `exam_year`, `question_id`, `modality`,
  `model_answer`, `correct_answer`, `raw_response`. It is NOT in .gitignore — it is
  the deliverable this PC produces and commits.
- Expected runtime on the 5070 Ti: ~15-30 min compute; the model download dominates.

## After the run

1. Spot-check: every row has a `model_answer`, `raw_response` is bare letters, row
   count = 532 (+ header).
2. `git add data/results.csv` -> commit as the user (plain message) -> `git push`.
3. The user pulls on the Mac and runs `python run_all.py` there (it also works here —
   scipy/statsmodels/matplotlib are in requirements.txt — but the Mac is the analysis
   machine).
