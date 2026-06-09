"""Generate per-question screenshot images for the 'screenshot' modality (design B).

For each type-B/C question we crop every Figure/Table it references and stitch
them vertically into data/screenshots/<Exam>_Q<N>.png. The question STEM is not
included — it is supplied as text by collect.py; the image carries only the
figure(s)/table(s), isolating the graph-reading variable.

Cropping dispatches by caption type:
  Figure -> union of vector drawings in the column (handles multi-panel figures)
  Table  -> whitespace-gap walk up from the caption (handles text/matrix tables)

Usage:
  python crop_figures.py <Exam>     # one exam
  python crop_figures.py ALL        # every data/encoded/*.json
"""

import json
import re
import sys
from pathlib import Path

import fitz
from PIL import Image

COLMID, ZOOM, GAP = 297.0, 2.0, 26.0
EXAMS = "ML-examsets"
OUT = Path("data/screenshots")
OUT.mkdir(parents=True, exist_ok=True)
cap_re = re.compile(r"^(Figure|Table)\s+(\d+)\b")
q_re = re.compile(r"^Question\s+\d+\b")
ref_re = re.compile(r"\b(Figure|Table)\s+(\d+)")


def col(x0):
    return 0 if x0 < COLMID else 1


def crop_exam_captions(doc):
    """Return {(kind,num): PIL.Image} for every caption in the exam PDF."""
    anchors = []
    for pno in range(len(doc)):
        for b in doc[pno].get_text("blocks"):
            head = " ".join(b[4].split())
            if q_re.match(head) or cap_re.match(head):
                anchors.append((pno, col(b[0]), b[3]))

    def clamp(reg, c):
        cx0, cx1 = (4, 296) if c == 0 else (299, 591)
        return fitz.Rect(max(reg.x0, cx0) - 4, reg.y0 - 5, min(reg.x1, cx1) + 4, reg.y1 + 5) & doc[0].rect

    def region_figure(pno, crect, c):
        tops = [y1 for (ap, ac, y1) in anchors if ap == pno and ac == c and y1 <= crect.y0 - 2]
        bt = max(tops) if tops else 24.0
        reg = fitz.Rect(crect)
        for d in doc[pno].get_drawings():
            r = d["rect"]
            cx = (r.x0 + r.x1) / 2
            if ((c == 0 and cx < COLMID) or (c == 1 and cx > COLMID)) and r.width > 1 and r.height > 1 \
                    and r.y0 >= bt - 2 and r.y1 <= crect.y0 + 2:
                reg |= r
        return clamp(reg, c)

    def region_table(pno, crect, c):
        items = []
        for b in doc[pno].get_text("blocks"):
            if b[4].strip() and col(b[0]) == c:
                head = " ".join(b[4].split())
                tag = "q" if q_re.match(head) else ("cap" if cap_re.match(head) else "txt")
                items.append((b[1], b[3], b[0], b[2], tag))
        for d in doc[pno].get_drawings():
            r = d["rect"]
            cx = (r.x0 + r.x1) / 2
            if r.width > 1 and r.height > 1 and ((c == 0 and cx < COLMID) or (c == 1 and cx > COLMID)):
                items.append((r.y0, r.y1, r.x0, r.x1, "draw"))
        reg, cur = fitz.Rect(crect), crect.y0
        while True:
            above = [it for it in items if it[1] <= cur - 0.5 and it[1] > 3]
            if not above:
                break
            it = max(above, key=lambda t: t[1])
            if it[4] in ("q", "cap") or cur - it[1] > GAP:
                break
            reg |= fitz.Rect(it[2], it[0], it[3], it[1])
            cur = min(cur, it[0])
        return clamp(reg, c)

    crops, seen = {}, set()
    for pno in range(len(doc)):
        for b in doc[pno].get_text("blocks"):
            m = cap_re.match(" ".join(b[4].split()))
            if not m or (m.group(1), int(m.group(2))) in seen:
                continue
            kind, num = m.group(1), int(m.group(2))
            seen.add((kind, num))
            crect, c = fitz.Rect(b[0], b[1], b[2], b[3]), col(b[0])
            reg = region_figure(pno, crect, c) if kind == "Figure" else region_table(pno, crect, c)
            pix = doc[pno].get_pixmap(clip=reg, matrix=fitz.Matrix(ZOOM, ZOOM))
            crops[(kind, num)] = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    # Fallback: captions merged into another block (e.g. glued after "E. Don't know.")
    # are missed above; find them at line granularity.
    for pno in range(len(doc)):
        for blk in doc[pno].get_text("dict")["blocks"]:
            for line in blk.get("lines", []):
                txt = "".join(s["text"] for s in line["spans"]).strip()
                m = cap_re.match(txt)
                if not m or (m.group(1), int(m.group(2))) in seen:
                    continue
                kind, num = m.group(1), int(m.group(2))
                seen.add((kind, num))
                crect = fitz.Rect(line["bbox"])
                c = col(crect.x0)
                reg = region_figure(pno, crect, c) if kind == "Figure" else region_table(pno, crect, c)
                pix = doc[pno].get_pixmap(clip=reg, matrix=fitz.Matrix(ZOOM, ZOOM))
                crops[(kind, num)] = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return crops


def vstack(imgs, gap=12):
    w = max(i.width for i in imgs)
    h = sum(i.height for i in imgs) + gap * (len(imgs) - 1)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    y = 0
    for im in imgs:
        canvas.paste(im, ((w - im.width) // 2, y))
        y += im.height + gap
    return canvas


def refs_in_order(text):
    out = []
    for kind, num in ref_re.findall(text or ""):
        key = (kind, int(num))
        if key not in out:
            out.append(key)
    return out


def process(exam):
    data = json.load(open(f"data/encoded/{exam}.json", encoding="utf-8"))
    doc = fitz.open(f"{EXAMS}/{exam}.pdf")
    crops = crop_exam_captions(doc)
    made, warns = 0, []
    for q in data["questions"]:
        if not q.get("needs_screenshot"):
            continue
        refs = refs_in_order(q["question_text"]) or refs_in_order(q.get("source_location", ""))
        imgs = [crops[r] for r in refs if r in crops]
        missing = [f"{k}{n}" for (k, n) in refs if (k, n) not in crops]
        if not imgs:
            warns.append(f"{q['question_id']}: no crops (refs={refs})")
            continue
        if missing:
            warns.append(f"{q['question_id']}: missing {missing} (used {[f'{k}{n}' for k,n in refs if (k,n) in crops]})")
        img = imgs[0] if len(imgs) == 1 else vstack(imgs)
        img.save(OUT / f"{exam}_{q['question_id']}.png")
        made += 1
    print(f"{exam}: wrote {made} screenshots" + (f"  | {len(warns)} warnings" if warns else ""))
    for w in warns:
        print(f"    ! {w}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "Spring2025"
    exams = [p.stem for p in sorted(Path("data/encoded").glob("*.json"))] if target == "ALL" else [target]
    for e in exams:
        process(e)
