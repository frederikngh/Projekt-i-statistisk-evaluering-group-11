# Minimal Gemma chat - run in the `gemma` env:  python simple_gemma.py
from transformers import pipeline

gemma = pipeline("image-text-to-text", model="google/gemma-4-E4B-it", dtype="auto", device="mps")

PROMPT = "Hvem er Frederik Naervig Gerlach-Hansen?"
out = gemma(text=[{"role": "user", "content": [{"type": "text", "text": PROMPT}]}], max_new_tokens=200)
print(out[0]["generated_text"][-1]["content"])
