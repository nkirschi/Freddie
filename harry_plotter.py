import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from itertools import chain, cycle
from matplotlib.dates import DateFormatter

LINEWIDTH = 0.25  # default plot linewidth

# update settings for LaTeX exporting
mpl.rcParams.update({
    "text.usetex": True,
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.latex.preamble": "\n".join([
        r"\usepackage{palatino}",
        r"\usepackage{siunitx}",
    ])
})


def plot_components_and_norm(x, Y, *, symbol, vlines=(), title=None, xlabel=None, ylabel=None, outfile=None):
    plt.gca().xaxis.set_major_formatter(DateFormatter("$%d$ %b '$%y$\n$%H$:$%M$", usetex=False))

    for (i, y) in enumerate(Y):
        plt.plot(x, y, linewidth=LINEWIDTH, label=f"${symbol}_{i + 1}$")

    n = np.linalg.norm(np.array(Y), axis=0)
    p = plt.plot(x, n, linewidth=LINEWIDTH, label=f"$\\pm||\\vec{{{symbol}}}||$")
    plt.plot(x, -n, linewidth=LINEWIDTH, color=p[-1].get_color())

    colors = ["cyan", "olive", "pink", "gray"]
    c = cycle(chain(colors, reversed(colors[:-1])))
    for i, x in enumerate(vlines[:-1]):
        plt.axvspan(x, vlines[i + 1], alpha=0.5, facecolor=next(c), edgecolor="black", linewidth=0)

    plt.grid(linewidth=LINEWIDTH)
    plt.margins(x=0)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(fontsize=8)
    plt.legend()

    if outfile:
        plt.savefig(outfile)
