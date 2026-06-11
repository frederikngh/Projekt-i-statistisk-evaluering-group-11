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
   the 5070 Ti is Blackwell/sm_120 and needs the cu128 build). DONE on this PC
   2026-06-10, with cu130 builds instead (base env had a corrupted mixed install —
   repaired; cu130 is fine for sm_120 with driver 610).
2. Hugging Face: the licence for `google/gemma-4-E2B-it` must be accepted on the model
   page (browser), then `hf auth login` once. Gated model — nothing downloads before
   this. DONE on this PC (model id is E2B since 2026-06-10, see "Known risks").
3. Smoke test: `python simple_gemma.py`. Must print `Using device: cuda`. First run
   downloads the model (~10 GB on disk).
4. Full run: `python collect.py` (~532 calls, prints `[i/532]`). Resume-safe: rows
   already in `data/results.csv` are skipped, so Ctrl+C + rerun is fine.
   Flags if needed: `--exam Fall2024`, `--modality text|screenshot|text_desc`, `--dry-run`.
   With the chain-of-thought protocol (2026-06-10) expect ~40-90 s per typical
   question and up to ~3 min for dense-arithmetic ones (~25 tok/s measured on E2B,
   max_new_tokens=4096), i.e. roughly 7-10 hours for all 532 — close LM Studio
   and other VRAM users first or it gets much slower.
5. Sanity-check `data/results.csv` (see below), then commit it as the user with a plain
   message like "Add Gemma results from the full run" and push.

## Known risks and the agreed responses

- **E4B was too slow — the agreed fallback IS NOW IN EFFECT (2026-06-10):** E4B bf16
  spilled out of the 16 GB VRAM (~4 tok/s with the CoT token budget), so per the agreed
  response the model is `google/gemma-4-E2B-it` for **ALL 532 calls** — one model
  consistently, never a mix. Model id changed in BOTH `collect.py` and `simple_gemma.py`;
  no E4B rows existed under the new protocol (E2B measured ~25 tok/s). If the model
  changes again: delete any partial `data/results.csv` first, change BOTH files, rerun
  everything, tell the user.
- **Parse canary (replaces the old "thinking-mode contamination" canary):** chain of
  thought is DELIBERATE since 2026-06-10 (user decision) — raw responses are SUPPOSED
  to be reasoning. The new invariant: every `raw_response` should END with a
  "FINAL ANSWER: X" line. Rows where that line is missing fall back to a tail-scan or
  E — if more than a handful of rows lack the line, STOP and investigate with the user.
- **Device says `cpu` or `mps` missing errors:** torch install went wrong; uninstall
  torch/torchvision and redo step 1. Do not run the collection on CPU.

## Hard rules for code changes

- **`build_prompt()` and `INSTRUCTION` in collect.py must stay byte-identical** to the
  CURRENT (2026-06-10) chain-of-thought baseline. The old single-letter prompt was
  deliberately replaced at the user's request ("i want chain of thought to be
  activated") — that change was authorized, one-time, and is the new invariant. Any
  further edit near the prompts must be proven harmless:
  `python collect.py --dry-run > after.txt` and diff against a pre-edit dump.
- Greedy decoding stays — reproducibility. Generation params MUST go through
  `generate_kwargs={"max_new_tokens": 4096, "do_sample": False}`: passed loose, the
  pipeline routes them to the processor which IGNORES them (found 2026-06-10 — the old
  short-answer run probably sampled instead of being greedy). The cap was 2048 for the
  first ~210 rows; 69 dense-arithmetic replies hit it mid-calculation (the FINAL ANSWER
  canary fired: 41 fake E rows + 24 unreliable tail-parses), so it was raised to 4096
  and the 69 truncated rows purged and re-collected. Finished replies are unaffected by
  a cap raise (greedy: the cap only binds when hit), so the kept rows stay valid.
- If you touch ANY project `.py`, match the deliberate 2nd-semester style: NO list
  comprehensions, no type hints, no pathlib/regex/Counter/defaultdict, plain loops +
  `append`, lists not tuples, string dict keys (`exam + " " + qid`), no chained calls
  on results (`result = binomtest(...)` then `result.pvalue`), no format specs —
  `round()` / `percent()` + string `+` concatenation, p-values via `fmt_p()`,
  copy-paste blocks fine, named matplotlib colors, plain ASCII.
  Exception: `crop_figures.py`, `build_questions_csv.py`, `build_screenshot_checklist.py`
  keep their more advanced style (one-time build tools).

## Repo facts

- `data/questions.csv` = the manifest collect.py reads: **532 rows = 134 `text` +
  271 `screenshot` + 127 `text_desc`** (15 exams x 27 questions, types A/B/C;
  counts changed in corrections batch 4 — Spring2023 Q25 became screenshot-only).
  Generated from `data/encoded/*.json` (the source of truth) — both are committed,
  so nothing needs regenerating on this PC.
- All 271 figure crops are committed in `data/screenshots/`; `questions.csv` uses
  relative paths, so **run everything from the repo root**.
- `data/results.csv` columns include `exam_year`, `question_id`, `modality`,
  `gemma_answer`, `correct_answer`, `raw_response`. It is NOT in .gitignore — it is
  the deliverable this PC produces and commits. `raw_response` now holds the full
  chain-of-thought reply (multi-line, csv-quoted), not a bare letter.
- `data/results_noCoT_archive.csv` = the 136 rows collected 2026-06-10 with the OLD
  single-letter prompt, E4B, broken stems, and the do_sample routing bug. Archived,
  NOT comparable with the current protocol, never to be mixed into results.csv.
- `data/results_preFix_archive.csv` = 179 CoT rows collected 2026-06-10 BEFORE
  corrections batch 4 (`CORRECTIONS_BATCH4.md`: ~70 stems de-embedded, wrong
  figures/digits fixed, all crops regenerated). Same protocol as current but stale
  question texts — never mix into results.csv.
- Expected runtime on the 5070 Ti with CoT on E2B: roughly 6-10 hours for all 532
  calls (~25 tok/s measured). The old "~15-30 min" figure was for the 16-token
  protocol and is obsolete.

## After the run

1. Spot-check: every row has a `gemma_answer`, (nearly) every `raw_response` ends
   with a "FINAL ANSWER: X" line (investigate if many do not), row count = 532
   (+ header).
2. `git add data/results.csv` -> commit as the user (plain message) -> `git push`.
3. The user pulls on the Mac and runs `python run_all.py` there (it also works here —
   scipy/statsmodels/matplotlib are in requirements.txt — but the Mac is the analysis
   machine).
