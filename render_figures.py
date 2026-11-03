"""Re-render the three submission figures with TrueType (Type 42) embedded fonts.

matplotlib's default pdf.fonttype is 3, which writes text as Type 3 glyph
procedures; Elsevier asks for embedded fonts / text as graphics, and Type 3 is
the form production departments most often reject. Setting fonttype 42 keeps
the PDF vector and makes the text searchable and embeddable.
"""
import runpy
import shutil

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

for script in ["paper_figure.py", "case_figure.py", "replot_fig3.py"]:
    runpy.run_path(script, run_name="__main__")

shutil.copyfile("fig1_combined.pdf", "Figure_1.pdf")
shutil.copyfile("fig2_screening.pdf", "Figure_2.pdf")
shutil.copyfile("fig1_combined.png", "Figure_1.png")
shutil.copyfile("fig2_screening.png", "Figure_2.png")
print("re-rendered Figure_1..3")
