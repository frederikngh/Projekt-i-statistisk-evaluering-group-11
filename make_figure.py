"""A quick bar chart: accuracy of the four question groups."""

import os

import matplotlib.pyplot as plt

from helpers import CHANCE, load_results, subset, text_A_rows, graph_only_rows
from helpers import count_correct

rows = load_results()
graph = graph_only_rows(rows)

# the image answers that are NOT pure-graph (i.e. the paired B/C ones)
screenshot_rows = subset(rows, "modality", "screenshot")
paired_image = []
for r in screenshot_rows:
    if r not in graph:
        paired_image.append(r)

groups = [text_A_rows(rows), subset(rows, "modality", "text_desc"), paired_image, graph]
labels = ["pure text\n(type A)", "table/figure\nas TEXT",
          "table/figure\nas IMAGE", "pure graph\n(image only)"]
colors = ["steelblue", "mediumseagreen", "darkorange", "indianred"]

plt.figure(figsize=(7, 4.5))
for i in range(4):
    if len(groups[i]) > 0:            # the example data has no pure-graph rows
        k = count_correct(groups[i])
        n = len(groups[i])
        plt.bar(i, k / n, color=colors[i], width=0.6)
        bar_label = str(round(100 * k / n)) + "%\nn=" + str(n)
        plt.text(i, k / n + 0.02, bar_label, ha="center", fontsize=9)
plt.axhline(CHANCE, linestyle="--", color="grey", label="chance (25%)")
plt.xticks([0, 1, 2, 3], labels, fontsize=9)
plt.ylim(0, 1.1)
plt.ylabel("accuracy")
plt.title("Gemma accuracy by question group", fontweight="bold")
plt.legend(fontsize=9)
plt.tight_layout()
if not os.path.exists("figures"):
    os.mkdir("figures")
plt.savefig("figures/accuracy_by_group.png", dpi=150)
print()
print("Figure saved: figures/accuracy_by_group.png")
