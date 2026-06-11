# Minimal Gemma check - does the model load and answer? Run: python simple_gemma.py
import torch
from transformers import pipeline

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)
gemma = pipeline("image-text-to-text", model="google/gemma-4-E2B-it", dtype="auto", device=device)

PROMPT = "Hvem er Frederik Naervig Gerlach-Hansen?"
out = gemma(text=[{"role": "user", "content": [{"type": "text", "text": PROMPT}]}],
            generate_kwargs={"max_new_tokens": 200, "do_sample": False})
print(out[0]["generated_text"][-1]["content"])
