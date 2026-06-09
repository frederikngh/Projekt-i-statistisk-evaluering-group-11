"""apply_corrections.py — apply the fixes found by the checker agents.

Surgical raw-text edits (scoped per question) so formatting stays byte-identical
apart from the intended changes. Re-runnable: it asserts the old value is present
before changing it, so a second run is a no-op-or-loud-failure.

Corrections (from the 15 independent checker agents):
  Fall2024  Q12 answer A -> B   (k-means cost = 50 = option B; worked solution says B)
  Fall2024  Q14 answer A -> D   (support(X) = 3/8 = option D)
  Fall2019  Q3  stem  x1=PM2.5 -> x1=MONTH   (x1 is MONTH per Table 1; PM2.5 is x2)
  Spring2023 Q7  figure ref "Figure 3" -> "Figure 2"  (exam numbering, not solution's)
  Spring2023 Q13 figure ref "Figure 4" -> "Figure 3"  (exam numbering)
"""

from pathlib import Path

ENC = Path("data/encoded")


def scoped_replace(text: str, qid: str, old: str, new: str) -> str:
    """Replace `old`->`new` only within the JSON span of question `qid`."""
    start = text.index(f'"question_id": "{qid}"')
    nxt = text.find('"question_id":', start + 20)
    end = nxt if nxt != -1 else len(text)
    span = text[start:end]
    assert old in span, f"{qid}: '{old}' not found in its span"
    return text[:start] + span.replace(old, new) + text[end:]


def patch(name: str, fn) -> None:
    p = ENC / f"{name}.json"
    p.write_text(fn(p.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"{name}: patched")


def fall2024(t: str) -> str:
    # answer_key block (top of file) — these substrings are unique to it
    assert t.count('"Q12": "A"') == 1 and t.count('"Q14": "A"') == 1
    t = t.replace('"Q12": "A"', '"Q12": "B"').replace('"Q14": "A"', '"Q14": "D"')
    # per-question correct_answer
    t = scoped_replace(t, "Q12", '"correct_answer": "A"', '"correct_answer": "B"')
    t = scoped_replace(t, "Q14", '"correct_answer": "A"', '"correct_answer": "D"')
    print("  Fall2024: Q12 A->B, Q14 A->D")
    return t


def fall2019(t: str) -> str:
    t = scoped_replace(t, "Q3", "x1=PM2.5", "x1=MONTH")
    print("  Fall2019: Q3 x1=PM2.5 -> x1=MONTH")
    return t


def spring2023(t: str) -> str:
    t = scoped_replace(t, "Q7", "Figure 3", "Figure 2")
    t = scoped_replace(t, "Q13", "Figure 4", "Figure 3")
    print("  Spring2023: Q7 Figure 3->2, Q13 Figure 4->3")
    return t


if __name__ == "__main__":
    patch("Fall2024", fall2024)
    patch("Fall2019", fall2019)
    patch("Spring2023", spring2023)
    print("All corrections applied.")
