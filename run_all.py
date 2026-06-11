"""Run the whole analysis: every test script + the figure, in order.

Usage:
    python run_all.py                    # uses data/results.csv
    python run_all.py data/example.csv   # try it on example data
"""

import os
import sys

arg = ""
if len(sys.argv) > 1:
    arg = ' "' + sys.argv[1] + '"'

scripts = ["binomial_test.py", "mcnemar_test.py", "text_vs_graph_test.py",
           "question_types_test.py", "dont_know_test.py", "power_check.py",
           "make_figure.py"]

for script in scripts:
    code = os.system("python " + script + arg)
    if code != 0:
        print()
        print("STOPPED:", script, "failed (exit code " + str(code) + ").")
        sys.exit(1)
