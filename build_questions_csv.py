"""build_questions_csv.py — generate data/questions.csv from data/encoded/*.json.

The encoded JSONs (data/encoded/<Exam>.json) are the format-neutral source of
truth. This converter emits the manifest that collect.py consumes. Re-run after
editing or adding any encoded JSON. If the target format changes, edit ONLY this
file — the encodings stay put.

Row rules per question:
  type A (pure text)      -> 1 row : modality=text
  type B/C (figure/table) -> 1 row : modality=screenshot (image_path set)
                          -> + 1 row modality=text_desc (description set)  IFF text_faithful
So geometric figures (text_faithful=false) are screenshot-only, by design.
"""

import csv
import glob
import json
import sys
from pathlib import Path

ENCODED = Path("data/encoded")
OUT = Path("data/questions.csv")
COLUMNS = [
    "exam_year", "question_id", "question_type", "topic",
    "modality", "question_text", "description", "image_path", "correct_answer",
]


def rows_for_question(exam: str, q: dict) -> list[dict]:
    base = {
        "exam_year": exam,
        "question_id": q["question_id"],
        "question_type": q["type"],
        "topic": q.get("topic", ""),
        "question_text": q["question_text"],
        "correct_answer": q["correct_answer"],
    }
    rows = []
    if q["type"] == "A":
        rows.append({**base, "modality": "text", "description": "", "image_path": ""})
    else:  # B or C
        rows.append({**base, "modality": "screenshot", "description": "",
                     "image_path": q.get("screenshot_path") or ""})
        if q.get("text_faithful") and q.get("text_representation"):
            rows.append({**base, "modality": "text_desc",
                         "description": q["text_representation"], "image_path": ""})
    return rows


def main():
    files = sorted(glob.glob(str(ENCODED / "*.json")))
    if not files:
        print("No encoded JSONs found in data/encoded/.")
        sys.exit(1)

    all_rows, per_exam = [], {}
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        exam = d["exam"]
        n_rows = 0
        for q in d["questions"]:
            rs = rows_for_question(exam, q)
            all_rows.extend(rs)
            n_rows += len(rs)
        per_exam[exam] = (len(d["questions"]), n_rows)

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(all_rows)

    by_mod = {}
    for r in all_rows:
        by_mod[r["modality"]] = by_mod.get(r["modality"], 0) + 1
    print(f"Wrote {OUT}: {len(all_rows)} rows from {len(files)} exams.")
    print("  by modality:", ", ".join(f"{k}={v}" for k, v in sorted(by_mod.items())))
    print("  per exam (questions -> rows):")
    for exam, (nq, nr) in sorted(per_exam.items()):
        print(f"    {exam:12s} {nq:3d} -> {nr:3d}")


if __name__ == "__main__":
    main()
