"""Structural validator for encoded exam JSONs (data/encoded/<Exam>.json).

Checks SCHEMA + internal consistency only (not content vs. the PDFs - the
answer keys were double-checked separately). Safe to run anytime.

    python validate_encoded.py                      # all data/encoded/*.json
    python validate_encoded.py data/encoded/Fall2024.json
"""

import glob
import json
import sys

REQUIRED_TOP = {"exam", "course", "n_questions", "answer_key", "questions"}
REQUIRED_Q = {
    "question_id", "number", "topic", "type", "text_faithful",
    "question_text", "options", "correct_answer",
    "text_representation", "needs_screenshot", "screenshot_path", "source_location",
}


def validate(path):
    errs = []
    d = json.load(open(path, encoding="utf-8"))

    for k in REQUIRED_TOP - set(d):
        errs.append(f"missing top-level key: {k}")

    qs = d.get("questions", [])
    if len(qs) != 27:
        errs.append(f"expected 27 questions, got {len(qs)}")

    numbers = []
    for q in qs:
        qid = q.get("question_id", "?")
        for k in REQUIRED_Q - set(q):
            errs.append(f"{qid}: missing field '{k}'")

        t = q.get("type")
        if t not in {"A", "B", "C"}:
            errs.append(f"{qid}: bad type {t!r}")

        if not isinstance(q.get("text_faithful"), bool):
            errs.append(f"{qid}: text_faithful must be bool")

        ca = q.get("correct_answer")
        if ca not in {"A", "B", "C", "D"}:
            errs.append(f"{qid}: correct_answer {ca!r} not in A-D (E is never the key)")

        ak = d.get("answer_key", {}).get(qid)
        if ak is not None and ak != ca:
            errs.append(f"{qid}: answer_key {ak!r} != correct_answer {ca!r}")

        if not {"A", "B", "C", "D"} <= set(q.get("options", {})):
            errs.append(f"{qid}: options missing some of A-D (has {sorted(q.get('options', {}))})")

        needs = q.get("needs_screenshot")
        spath = q.get("screenshot_path")
        tr = q.get("text_representation")
        tf = q.get("text_faithful")
        if t == "A":
            if needs:
                errs.append(f"{qid}: type A but needs_screenshot is true")
            if spath:
                errs.append(f"{qid}: type A but screenshot_path is set")
            if tr:
                errs.append(f"{qid}: type A but text_representation is set")
        elif t in {"B", "C"}:
            if not needs:
                errs.append(f"{qid}: type {t} but needs_screenshot is false")
            if not spath:
                errs.append(f"{qid}: type {t} but screenshot_path is empty")
            if tf and not tr:
                errs.append(f"{qid}: text_faithful true but no text_representation")
            if (not tf) and tr:
                errs.append(f"{qid}: text_faithful false but text_representation is set")

        numbers.append(q.get("number"))

    if numbers != list(range(1, len(qs) + 1)):
        errs.append(f"'number' fields are not 1..{len(qs)} in order")

    return errs, d


def main():
    paths = sys.argv[1:] or sorted(glob.glob("data/encoded/*.json"))
    if not paths:
        print("No JSONs found in data/encoded/.")
        sys.exit(0)

    total = 0
    for p in paths:
        try:
            errs, d = validate(p)
        except Exception as e:
            print(f"X  {p}: FAILED TO PARSE: {e}")
            total += 1
            continue
        if errs:
            total += len(errs)
            print(f"X  {p}: {len(errs)} issue(s)")
            for e in errs:
                print(f"     - {e}")
        else:
            c = {"A": 0, "B": 0, "C": 0}
            for q in d["questions"]:
                c[q["type"]] = c.get(q["type"], 0) + 1
            faithful_b = sum(1 for q in d["questions"] if q["type"] == "B" and q["text_faithful"])
            print(f"OK {p}: 27 Qs  (A={c['A']} B={c['B']} C={c['C']}; "
                  f"describable B={faithful_b})")

    print(f"\n{'ALL OK' if total == 0 else f'{total} total issue(s)'}")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
