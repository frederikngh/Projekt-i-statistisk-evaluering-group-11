"""build_screenshot_checklist.py — list every screenshot to capture, from data/encoded/*.json.

Writes SCREENSHOTS.md: a checkbox list grouped by exam of the type-B/C questions
whose `screenshot_path` you must fill. Screenshot the WHOLE question (stem +
options + its figure/table) from the exam PDF into the listed path. Type-A
questions need no screenshot.
"""

import glob
import json
from pathlib import Path

OUT = Path("SCREENSHOTS.md")


def main():
    files = sorted(glob.glob("data/encoded/*.json"))
    if not files:
        print("No encoded JSONs found in data/encoded/.")
        return

    body, total = [], 0
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        shots = [q for q in d["questions"] if q.get("needs_screenshot")]
        body.append(f"## {d['exam']}  ({len(shots)} screenshots)\n")
        for q in shots:
            total += 1
            body.append(
                f"- [ ] `{q['screenshot_path']}` — {q['question_id']} "
                f"[type {q['type']}] {q.get('source_location', '')}"
            )
        body.append("")

    header = [
        "# Screenshot checklist",
        "",
        "One screenshot per type-B/C question: capture the WHOLE question (stem + options +",
        "its figure/table) from the exam PDF and save it to the listed path. Type-A questions",
        "need none.",
        "",
        f"**Total: {total} screenshots across {len(files)} exams.**",
        "",
    ]
    OUT.write_text("\n".join(header + body), encoding="utf-8")
    print(f"Wrote {OUT}: {total} screenshots across {len(files)} exams.")


if __name__ == "__main__":
    main()
