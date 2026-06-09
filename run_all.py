"""Run the whole analysis: every test script + the figure, in order.

Usage:
    python run_all.py                    # uses data/results.csv
    python run_all.py data/example.csv   # try it on example data
"""

import os
import sys

arg = ""
if len(sys.argv) > 1:
    arg = " " + sys.argv[1]

os.system("python binomial_test.py" + arg)
os.system("python mcnemar_test.py" + arg)
os.system("python text_vs_graph_test.py" + arg)
os.system("python question_types_test.py" + arg)
os.system("python dont_know_test.py" + arg)
os.system("python power_check.py" + arg)
os.system("python make_figure.py" + arg)
