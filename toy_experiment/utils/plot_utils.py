""" Code copied and modified from
https://github.com/dlwhittenbury/von-Mises-Sampling/blob/master/von-Mises.ipynb
"""

from collections import Counter
from pathlib import Path
from warnings import warn
from typing import Dict, Union, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import seaborn as sns
import mlflow as mlf

from training import Trainer
from data import LiftingDist1Dto2D
from utils.utils import polar2cartesian


PAGE_WIDTH_IN = 6.875
COL_WIDTH_IN = 3.25
FONTSIZE = 8


def setup_style(grid=False, column_fig=False):
    plt.style.use("seaborn-paper")

    if column_fig:
        plt.rcParams["figure.figsize"] = (COL_WIDTH_IN, COL_WIDTH_IN / 2)
    else:
        plt.rcParams["figure.figsize"] = (PAGE_WIDTH_IN, PAGE_WIDTH_IN / 2)
    plt.rcParams["axes.grid"] = grid
    # lw = 1.0 if column_fig else 0.5
    plt.rcParams.update(
        {
            "font.size": FONTSIZE,
            "legend.fontsize": "x-small",
            "axes.labelsize": "small",
            "xtick.labelsize": "small",
            "ytick.labelsize": "small",
            "axes.titlesize": "medium",
            "lines.linewidth": 0.5,
        }
    )


def plot_circle(
    r: float = 1.,
    axes: str = "off",
    fs: float = FONTSIZE,
    ax: Optional[plt.Axes] = None,
    show_ticks: bool = False,
    show_center: bool = False,
    show_input_ax: bool = True,
    show_input_ax_label: bool = False,
    inputs_offset: float = 2.,
    show_output_axs: bool = False,
):
    # Angular grid
    ctheta = np.linspace(0, 2 * np.pi, 500, endpoint=False)

    # Convert polar coordinates to cartesian coordinates
    [x, y] = polar2cartesian(r, ctheta)

    # Plot the unit circle
    if ax is None:
        ax = plt.gca()
    ax.plot(x, y, color="grey", lw=2, zorder=0, ls="--")

    # Layout
    ax.set_aspect("equal")
    ax.axis(axes)

    # Axis center
    if show_center:
        ax.scatter(0, 0, marker="+", s=50, color="black")
    
    if show_ticks:
        #  Add axes angles in radians
        ax.text(0.8, -0.05, r"$0$", fontsize=fs)
        ax.text(-0.17, 0.77, r"$\pi/2$", fontsize=fs)
        ax.text(-0.87, -0.05, r"$\pi$", fontsize=fs)
        ax.text(-0.17, -0.85, r"$-\pi/2$", fontsize=fs)

        # Add axes tick marks
        ax.scatter(1, 0, marker="+", s=50, color="black")
        ax.scatter(0, 1, marker="+", s=50, color="black")
        ax.scatter(-1, 0, marker="+", s=50, color="black")
        ax.scatter(0, -1, marker="+", s=50, color="black")

    # plot inputs axis
    if show_input_ax:
        ax.arrow(
            -1.2 * r, -inputs_offset,
            2.4 * r, 0.,
            width=0.01,
            head_width=0.1,
            edgecolor=None,
            facecolor="black",
        )
        if show_input_ax_label:
            ax.text(1.2 * r, -0.2 * r - inputs_offset, s="x")

    # plot both outputs axis
    if show_output_axs:
        ax.arrow(
            0., 0.,
            1.2 * r, 0.,
            width=0.01,
            head_width=0.1,
            edgecolor=None,
            facecolor="black",
        )
        ax.text(1.2 * r, -0.2 * r, s="x")

        ax.arrow(
            0., 0.,
            0., 1.2 * r,
            width=0.01,
            head_width=0.1,
            edgecolor=None,
            facecolor="black",
        )
        ax.text(-0.2 * r, 1.2 * r, s="y")

    return ax


def plot_angular_density(
    theta: np.array,
    pdf: np.array,
    r: float = 1.0,
    colour: str = "blue",
    maxline: bool = False,
    axes: str = "off",
    fs: float = FONTSIZE,
    ax: plt.Axes = None,
    show_ticks: bool = False,
    show_center: bool = False,
    label: Optional[str] = None,
    inputs_offset: float = 2.0,
):

    """
    plot_angular_density(theta,pdf,colour="blue",maxline=False,axes="off",fs=16)
    ============================================================================

    Plots the probability density function of a circular distribution on the
    unit circle.

    INPUT:

        * theta - angular grid - an array of floats.
        * pdf - the values of the probability density function on the angular
          grid theta.
        * r - radius
        * colour - an optional argument, the colour of the pdf curve, a string.
        * maxline - an optional argument, whether or not to include a line
          connecting the centre of the circle with the maximum value of pdf,
          boolean.
        * axes - an optional argument, whether or not to include the axes,
          boolean.
        * fs - an optional argument, the fontsize.

    OUTPUT:

        * A plot on a circle of a circular distribution.
        * ax (axes) - axes on which the plot is constructed.

    """

    # Draw the unit circle
    # ====================

    ax = plot_circle(
        r=r,
        axes=axes,
        fs=fs,
        ax=ax,
        show_ticks=show_ticks,
        show_center=show_center,
        inputs_offset=inputs_offset,
    )

    # Draw angular probability density
    # ================================

    # PDF will be drawn on the circle, so we need to account for the radius of
    # the unit circle
    d = r + pdf

    # Convert polar coordinates of the pdf to cartesian coordinates
    [xi, yi] = polar2cartesian(d, theta)

    # Plot the PDF
    ax.plot(xi, yi, color=colour, lw=2, label=label, zorder=1)

    if maxline:
        # Add a line from the origin to the maximum of the pdf
        idx = np.argmax(pdf)
        lx = [0, xi[idx]]
        ly = [0, yi[idx]]
        plt.plot(lx, ly, color="black", alpha=0.3)

    return ax


def distinct_colours(labels):

    """
    distinct_colours(labels)
    ========================

    INPUT:

        * labels - an array of labels.

    OUTPUT:

        * dic - a dictionary with keys equal to the distinct classes in
        labels and the values are the distinct colours.

    NOTES: requires "from collections import Counter"

    """

    # keys = labels, values = counts
    dic = dict(Counter(labels))

    # list of distinct labels
    distinct_labels = list(dic.keys())

    # number of distinct labels
    num_labels = len(distinct_labels)

    # distinct colours, length is num_labels
    distinct_colours = cm.rainbow(np.linspace(0, 1, num_labels))

    # Replace counts with colours in the dictionary
    for i, l in enumerate(dic):
        dic[l] = distinct_colours[i]

    return dic


def plot_angular_scatter(
    samples, labels=None, r=1.0, ms=100, axes="off", ax=None
):

    """
    plot_angular_scatter(samples,labels=None,ms=100)
    ================================================

    Creates a scatter plot on a circle of samples from a circular distribution.

    INPUT:

        * samples - samples of a circular distribution - either a scalar or
          array of floats.
        * labels - an optional argument containing class labels, default is
          None.
        * r - radius
        * ms - an optional argument markersize, default is 100.

    OUTPUT:

        * Scatter plot on a circle of samples from a circular distribution.
        * ax (axes) - axes on which the plot is constructed.

    """

    # Plot the unit circle S^{1}
    # ==========================

    # Angular grid
    theta = np.linspace(0, 2 * np.pi, 500, endpoint=False)

    # Convert polar coordinates to cartesian coordinates
    [x, y] = polar2cartesian(r, theta)

    # Plot the circle
    if ax is None:
        ax = plt.gca()
    ax.plot(
        x, y, color="black", lw=2, zorder=0
    )  # zorder=0 puts the circle behind all the data

    # Make aspect square
    ax.set_aspect("equal")

    # Make the scatter plot of samples
    # ================================

    # Convert polar coordinates to cartesian coordinates
    [xi, yi] = polar2cartesian(r, samples)

    if labels is None:
        ax.scatter(xi, yi, label=labels)
    else:

        # if labels are given create a dictionary were the keys are
        # the distinct labels and the values are distinct colours
        dic = distinct_colours(labels)

        # Loop over each distinct class (label)
        for k in dic.keys():

            # Get indices for this distinct class
            indices = [i for i, label in enumerate(labels) if label == k]

            # Get coordinates for this distinct class
            x = [xi[i] for i in indices]
            y = [yi[i] for i in indices]

            # Plot this class
            ax.scatter(x, y, color=dic[k], label=k, s=ms)

    # Turn axes off
    ax.axis(axes)

    return ax


def save_fig_and_log(fig, save_path, log_to_mlf):
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    if log_to_mlf:
        if save_path is None:
            warn(
                "Cannot log to MLFlow unless the figure is saved somewhere."
                "Try again with save_path not equal to None.",
                UserWarning,
            )
        else:
            mlf.log_artifact(save_path)


def plot_training_curve(
    trainer: Trainer,
    save_path: Union[str, Path] = None,
    log_to_mlf: bool = False,
):
    fig, ax = plt.subplots(figsize=(COL_WIDTH_IN, COL_WIDTH_IN // 2))
    setup_style()

    n_epochs = len(trainer.loss_list)
    epochs = np.arange(1, n_epochs + 1)

    ax.plot(
        epochs,
        trainer.loss_list,
        label="Training loss",
    )

    ax.plot(
        epochs,
        trainer.val_loss_list,
        label="Validation loss",
    )

    ax.legend(loc=0)
    ax.grid(True)

    save_fig_and_log(fig, save_path, log_to_mlf)


def plot_predictions(
    distribution: LiftingDist1Dto2D,
    X_test: np.array,
    Y_test: np.array,
    predictions_dict: Dict[str, np.array],
    offset: float = 2.0,
    palette: ListedColormap = sns.color_palette("muted"),
    save_path: Union[str, Path] = None,
    log_to_mlf: bool = False,
):
    fig, ax = plt.subplots(figsize=(COL_WIDTH_IN, COL_WIDTH_IN))
    setup_style()

    # plot circle and underlying distribution
    t = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    pdf = distribution.pdf(t)
    plot_angular_density(
        t, pdf, r=distribution.radius, colour="green", axes="off", ax=ax
    )

    # plot inputs
    ax.scatter(
        X_test,
        np.ones_like(X_test) * -offset,
        label="test inputs",
        alpha=0.6,
        c=palette[1],
    )

    # plot GT
    ax.scatter(
        Y_test[:, 0], Y_test[:, 1], marker="x", label="GT", c=palette[1]
    )

    # plot predictions
    for i, (method, predictions) in enumerate(predictions_dict.items()):
        ax.scatter(
            predictions[:, 0],
            predictions[:, 1],
            marker="o",
            label=method,
            c=palette[2 + i],
            alpha=0.6,
        )
    plt.legend(loc=0)
    plt.tight_layout()

    save_fig_and_log(fig, save_path, log_to_mlf)
