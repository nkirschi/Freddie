import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.dates import DateFormatter


# update settings for LaTeX exporting
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False
})


def plot_components_and_norm(x, Y, *, symbol, vlines=()):
    plt.gca().xaxis.set_major_formatter(DateFormatter("%H:%M"))

    for (i, y) in enumerate(Y):
        plt.plot(x, y, linewidth=0.25, label=f"${symbol}_{i}$")

    n = np.linalg.norm(np.array(Y), axis=0)
    p = plt.plot(x, n, linewidth=0.25, label=f"$\\pm||\\vec{{{symbol}}}||$")
    plt.plot(x, -n, linewidth=0.25, color=p[-1].get_color())

    for x in vlines:
        plt.axvline(x, linewidth=0.25, color="#000000")

    plt.legend()