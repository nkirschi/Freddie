import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def configure_latex_rendering():
    # mpl.use("pgf")
    mpl.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False
    })


def plot_components_and_norm(X, Y, Z, *, varname):
    plt.figure()

    plt.plot(X, linewidth=0.25, label=f"${varname}_1$")
    plt.plot(Y, linewidth=0.25, label=f"${varname}_2$")
    plt.plot(Z, linewidth=0.25, label=f"${varname}_3$")

    N = np.linalg.norm(np.array([X, Y, Z]), axis=0)
    p = plt.plot(N, linewidth=0.25, label=f"$\\pm||\\vec{{{varname}}}||$")
    plt.plot(-N, linewidth=0.25, color=p[-1].get_color())

    plt.legend()