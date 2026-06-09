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
    "Respond with ONLY a single capital letter: A, B, C, D, or E. "
    "Do not explain your answer."
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
        model="google/gemma-4-E4B-it",
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

    out = model(text=messages, max_new_tokens=16, do_sample=False)
    reply = out[0]["generated_text"][-1]["content"]
    raw = reply.strip()
    return extract_letter(raw), raw


def extract_letter(text):
    """Find the answer letter in Gemma's reply (usually the reply IS just 'B').

    Handles replies like 'B', 'B.', 'Answer: B'. If no letter is found we
    count it as E ('don't know') - the raw reply is saved so we can check.
    """
    text = text.upper()
    punctuation = [".", ",", ":", ";", "!", "(", ")", "*", "'", '"']
    for ch in punctuation:
        text = text.replace(ch, " ")
    for word in text.split():
        if word in ["A", "B", "C", "D", "E"]:
            return word
    return "E"


# ---------- Prompts ----------

def build_prompt(row):
    """Return (prompt_text, image_path or None) for one question row."""
    question = row["question_text"].strip()

    if row["modality"] == "text":
        prompt = (f"{INSTRUCTION}\n\n"
                  f"Question:\n{question}\n\n"
                  "Your answer (A/B/C/D/E):")
        return prompt, None

    if row["modality"] == "screenshot":
        prompt = (f"{INSTRUCTION}\n\n"
                  f"Question:\n{question}\n\n"
                  "The figure(s)/table(s) this question refers to are in the attached image.\n\n"
                  "Your answer (A/B/C/D/E):")
        return prompt, row["image_path"]

    # modality == "text_desc": the figure/table is written out as text instead
    prompt = (f"{INSTRUCTION}\n\n"
              "The question includes a figure/table that has been described in text below.\n"
              "Use this description to answer the question.\n\n"
              f"{question}\n\n"
              f"Figure/Table Description:\n{row['description']}\n\n"
              "Your answer (A/B/C/D/E):")
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
    new_file = not os.path.exists(RESULTS)
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
