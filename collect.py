"""collect.py - ask Gemma every exam question and save its answers.

Reads data/questions.csv (one row per question+modality), builds a prompt
for each row, sends it to Gemma, and appends the answer to data/results.csv.

Usage:
    python collect.py                      # run everything not done yet
    python collect.py --modality text      # only one modality
    python collect.py --exam Fall2024      # only one exam
    python collect.py --dry-run            # print prompts, don't run Gemma

Rows already in results.csv are skipped, so the script is safe to stop
and rerun. Every question is sent as a NEW call with no chat history
(the answers must be independent for our statistics to be valid), and
greedy decoding (do_sample=False) makes the answers reproducible.

Gemma is asked to reason step by step (chain of thought) and to finish
with a line "FINAL ANSWER: X"; extract_letter() parses that line from
the END of the reply, so option letters mentioned during the reasoning
are not picked up by mistake.
"""

import argparse
import csv
import os

QUESTIONS = "data/questions.csv"
RESULTS = "data/results.csv"

COLUMNS = ["exam_year", "question_id", "question_type", "topic",
           "modality", "gemma_answer", "correct_answer", "raw_response"]

INSTRUCTION = (
    "You are answering a multiple-choice exam question from a university "
    "Machine Learning course. Each question has exactly one correct answer "
    "among A, B, C, D. You may also answer E if you genuinely do not know. "
    "Think the problem through step by step: write out your reasoning and "
    "any calculations first, but keep each step short and to the point. "
    "ALWAYS finish your reply with one line of exactly this form:\n"
    "FINAL ANSWER: X\n"
    "where X is a single capital letter A, B, C, D, or E."
)


# ---------- Gemma ----------

def pick_device():
    """Use the NVIDIA GPU if there is one, otherwise Apple Silicon, otherwise CPU."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_gemma():
    """Load Gemma once (takes a while the first time)."""
    from transformers import pipeline
    device = pick_device()
    print("Loading Gemma on " + device + "...", flush=True)
    model = pipeline(
        "image-text-to-text",
        model="google/gemma-4-E2B-it",
        dtype="auto",
        device=device,
    )
    print("Gemma loaded.", flush=True)
    return model


def query_gemma(model, prompt, image_path):
    """Send ONE isolated question to Gemma. Returns (letter, raw_response)."""
    content = []
    if image_path:
        content.append({"type": "image", "url": os.path.abspath(image_path)})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    # 4096 new tokens leaves room for the step-by-step reasoning (dense
    # calculation questions hit a 2048 cap mid-arithmetic; the cap only
    # matters when it is hit, so finished replies are unaffected). Greedy
    # decoding keeps the answers reproducible. The params must go through
    # generate_kwargs: passed loose, the pipeline hands them to the
    # processor, which ignores them (do_sample would silently not apply).
    out = model(text=messages,
                generate_kwargs={"max_new_tokens": 4096, "do_sample": False})
    reply = out[0]["generated_text"][-1]["content"]
    raw = reply.strip()
    return extract_letter(raw), raw


def extract_letter(text):
    """Find the FINAL ANSWER letter in Gemma's (chain-of-thought) reply.

    The prompt asks Gemma to finish with a line like "FINAL ANSWER: B". We
    look at the LAST "FINAL ANSWER" in the reply, so option letters that come
    up during the reasoning are not picked up by mistake. If that line is
    missing (for example the reply was cut off at the token limit) we accept
    a line in the last three non-empty lines that IS just a letter (maybe
    decorated, like "(B)" or "**B.**"). A letter inside a sentence does NOT
    count: in upper-case text the word "A" is usually just the article "a",
    so a cut-off reply must not be scored as the answer A by accident. If
    nothing is found we count it as E ('don't know') - the raw reply is
    saved in results.csv so we can check those rows by hand.
    """
    letters = ["A", "B", "C", "D", "E"]
    skip = [" ", ":", "-", "*", "(", ")", ".", "\t"]
    text = text.upper()

    # 1) the FINAL ANSWER line (take the LAST one in the reply)
    pos = text.rfind("FINAL ANSWER")
    if pos != -1:
        rest = text[pos + len("FINAL ANSWER"):]
        i = 0
        while i < len(rest) and rest[i] in skip:
            i = i + 1
        if i < len(rest) and rest[i] in letters:
            # make sure it is a single letter, not the start of a word
            if i + 1 == len(rest) or not rest[i + 1].isalnum():
                return rest[i]

    # 2) fallback: a line that IS just the answer letter, nothing else
    tail = []
    for line in text.splitlines():
        line = line.strip()
        if line != "":
            tail.append(line)
    tail = tail[-3:]
    decorations = [".", ",", ":", ";", "!", "?", "(", ")", "*", "'", '"',
                   "[", "]", " ", "\t"]
    i = len(tail) - 1
    while i >= 0:
        line = tail[i]
        for ch in decorations:
            line = line.replace(ch, "")
        if line in letters:
            return line
        i = i - 1
    return "E"


# ---------- Prompts ----------

def build_prompt(row):
    """Return (prompt_text, image_path or None) for one question row."""
    question = row["question_text"].strip()

    if row["modality"] == "text":
        prompt = (f"{INSTRUCTION}\n\n"
                  f"Question:\n{question}\n\n"
                  "Reason step by step, then end with your FINAL ANSWER line.")
        return prompt, None

    if row["modality"] == "screenshot":
        prompt = (f"{INSTRUCTION}\n\n"
                  f"Question:\n{question}\n\n"
                  "The figure(s)/table(s) this question refers to are in the attached image.\n\n"
                  "Reason step by step, then end with your FINAL ANSWER line.")
        return prompt, row["image_path"]

    # modality == "text_desc": the figure/table is written out as text instead
    prompt = (f"{INSTRUCTION}\n\n"
              "The question includes a figure/table that has been described in text below.\n"
              "Use this description to answer the question.\n\n"
              f"{question}\n\n"
              f"Figure/Table Description:\n{row['description']}\n\n"
              "Reason step by step, then end with your FINAL ANSWER line.")
    return prompt, None


# ---------- Results file ----------

def load_done():
    """The exam+question+modality combinations already answered."""
    done = []
    if os.path.exists(RESULTS):
        with open(RESULTS, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = row["exam_year"] + " " + row["question_id"] + " " + row["modality"]
                done.append(key)
    return done


def append_result(result):
    """Add one answer to results.csv (creating the file if needed)."""
    new_file = not os.path.exists(RESULTS) or os.path.getsize(RESULTS) == 0
    with open(RESULTS, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(result)


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exam", help="only this exam, e.g. Fall2024")
    parser.add_argument("--modality", help="text / screenshot / text_desc")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the prompts instead of running Gemma")
    args = parser.parse_args()

    # read the manifest, applying the filters
    questions = []
    with open(QUESTIONS, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if args.exam and row["exam_year"] != args.exam:
                continue
            if args.modality and row["modality"] != args.modality:
                continue
            questions.append(row)

    # skip what we already have
    done = load_done()
    pending = []
    for q in questions:
        key = q["exam_year"] + " " + q["question_id"] + " " + q["modality"]
        if key not in done:
            pending.append(q)

    # safety: every screenshot row must have its image file on disk
    for q in pending:
        if q["modality"] == "screenshot":
            if q["image_path"] == "" or not os.path.exists(q["image_path"]):
                print("ERROR: missing screenshot for", q["exam_year"],
                      q["question_id"], "->", q["image_path"])
                return

    print("Questions selected :", len(questions))
    print("Already answered   :", len(done))
    print("Still to do        :", len(pending))

    if len(pending) == 0:
        print("Nothing to do. Run run_all.py to see the results.")
        return

    if args.dry_run:
        for row in pending:
            prompt, image = build_prompt(row)
            print("\n----", row["exam_year"], row["question_id"], row["modality"], "----")
            print(prompt[:400])
            if image:
                print("[image:", image, "]")
        return

    model = load_gemma()
    i = 0
    for row in pending:
        i = i + 1
        print("\n[" + str(i) + "/" + str(len(pending)) + "] "
              + row["exam_year"] + " " + row["question_id"] + " (" + row["modality"] + ")",
              flush=True)
        prompt, image = build_prompt(row)
        answer, raw = query_gemma(model, prompt, image)
        correct = row["correct_answer"].strip().upper()
        if answer == correct:
            verdict = "right"
        else:
            verdict = "wrong"
        print("  Gemma:", answer, "  correct answer:", correct, "  ->", verdict, flush=True)
        append_result({
            "exam_year": row["exam_year"],
            "question_id": row["question_id"],
            "question_type": row["question_type"],
            "topic": row["topic"],
            "modality": row["modality"],
            "gemma_answer": answer,
            "correct_answer": correct,
            "raw_response": raw,
        })

    print("\nDone. Results saved to", RESULTS)
    print("Next step: python run_all.py")


if __name__ == "__main__":
    main()
