import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.dates import DateFormatter

LINEWIDTH = 0.25  # default plot linewidth

# update settings for LaTeX exporting
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False
})


def plot_components_and_norm(x, Y, *, symbol, vlines=(), outfile=None):
    plt.gca().xaxis.set_major_formatter(DateFormatter("%H\\mathord{:}%M"))

    for (i, y) in enumerate(Y):
        plt.plot(x, y, linewidth=LINEWIDTH, label=f"${symbol}_{i + 1}$")

    n = np.linalg.norm(np.array(Y), axis=0)
    p = plt.plot(x, n, linewidth=LINEWIDTH, label=f"$\\pm||\\vec{{{symbol}}}||$")
    plt.plot(x, -n, linewidth=LINEWIDTH, color=p[-1].get_color())

    for x in vlines:
        plt.axvline(x, linewidth=LINEWIDTH, color="black")

    plt.legend()

    if outfile:
        plt.savefig(outfile)