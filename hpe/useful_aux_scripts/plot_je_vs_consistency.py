# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# %%
methods = [
    {
        "method": "MixSTE",
        "MPJPE": 40.9,
        "MPSCE": 9.9,
        "MPSSE": 8.8,
        "Hypotheses_N": 1,
        "Hypotheses": "1",
        "Constraints": "unconstrained",
        "ours": False,
    },
    {
        "method": "PoseFormer",
        "MPJPE": 44.3,
        "MPSCE": 7.2,
        "MPSSE": 4.3,
        "Hypotheses_N": 1,
        "Hypotheses": "1",
        "Constraints": "unconstrained",
        "ours": False,
    },
    {
        "method": "VideoPose3D",
        "MPJPE": 46.8,
        "MPSCE": 7.8,
        "MPSSE": 6.5,
        "Hypotheses_N": 1,
        "Hypotheses": "1",
        "Constraints": "unconstrained",
        "ours": False,
    },
    {
        "method": "ST-GCN",
        "MPJPE": 48.8,
        "MPSCE": 10.8,
        "MPSSE": 8.9,
        "Hypotheses_N": 1,
        "Hypotheses": "1",
        "Constraints": "unconstrained",
        "ours": False,
    },
    {
        "method": "\nMixSTE w/ MR",
        "MPJPE": 42.3,
        "MPSCE": 7.3,
        "MPSSE": 5.7,
        "Hypotheses_N": 1,
        "Hypotheses": "1",
        # "Constraints": "regularized",
        "Constraints": "unconstrained",
        "ours": False,
    },
    {
        "method": "ManiPose w/o MH",
        "MPJPE": 44.6,
        "MPSCE": 0.5,
        # "MPSCE": 0.7,
        "MPSSE": 0.3,
        "Hypotheses_N": 1,
        "Hypotheses": "1",
        "Constraints": "constrained",
        "ours": False,
    },
    {
        "method": "Anatomy3D",
        "MPJPE": 44.1,
        "MPSCE": 2.0,
        "MPSSE": 1.4,
        "Hypotheses_N": 1,
        "Hypotheses": "1",
        "Constraints": "constrained",
        "ours": False,
    },
    {
        "method": "MHFormer",
        "MPJPE": 43.0,
        "MPSCE": 8.0,
        "MPSSE": 5.7,
        "Hypotheses_N": 3,
        "Hypotheses": "3–5",
        "Constraints": "unconstrained",
        "ours": False,
    },
    {
        "method": "ManiPose (ours)",
        # "method": "ManiPose\n(ours) - P-Best",
        "MPJPE": 39.1,
        # "MPJPE": 41.9,
        "MPSCE": 0.5,
        # "MPSCE": 0.7,
        "MPSSE": 0.3,
        "Hypotheses_N": 5,
        "Hypotheses": "3–5",
        "Constraints": r"constrained",
        "ours": True,
    },
    {
        "method": "Sharma et. al.",
        "MPJPE": 46.8,
        "MPSCE": 9.9,
        "MPSSE": 13.0,
        "Hypotheses_N": 10,
        "Hypotheses": "10–20",
        "Constraints": "unconstrained",
        "ours": False,
    },
    {
        "method": "D3DP",
        # "method": "D3DP - P-Best",
        "MPJPE": 39.5,
        "MPSCE": 9.0,
        "MPSSE": 6.9,
        "Hypotheses_N": 20,
        "Hypotheses": "10–20",
        "Constraints": "unconstrained",
        "ours": False,
    },
    # {
    #     "method": "D3DP - J-Best",
    #     "MPJPE": 35.4,
    #     "MPSCE": 8.8,
    #     "MPSSE": 6.8,
    #     "Hypotheses": 340,
    #     "Constraints": "unconstrained",
    #     "ours": False,
    # },
    # {
    #     "method": "ManiPose\n(ours) - J-Best",
    #     "MPJPE": 36.7,
    #     "MPSCE": 0.7,
    #     "MPSSE": 0.3,
    #     "Hypotheses": 85,
    #     "Constraints": r"constrained",
    #     "ours": True,
    # },
    {
        "method": "Wehrbein et. al.",
        # "method": "Wehrbein et. al.\n(NF-based)",
        "MPJPE": 44.3,
        "MPSCE": 14.8,
        "MPSSE": 12.2,
        "Hypotheses_N": 200,
        "Hypotheses": "200",
        "Constraints": "unconstrained",
        "ours": False,
    },
]

methods = pd.DataFrame(methods)

# %%
sns.set_theme(style="whitegrid")

# %%

def label_points(x, y, size, val, ax):
    a = pd.concat({'x': x, 'y': y, 'size': (1/72)*np.sqrt(size), 'val': val}, axis=1)
    for _, point in a.iterrows():
        x = point['x'] + point['size'] + .1
        y = point['y'] - point['size'] + .02
        weight = "normal"

        if "w/ MR" in point['val']:
            y = point['y'] - point['size'] - 1.6
            x = point['x'] - point['size'] - 1.
        elif point['val'] == "ST-GCN":
            y = point['y'] + 0.5
            x = point['x'] - 1

        if "ManiPose (ours)" in point['val']:
            weight = "bold"

        ax.text(
            x,
            y,
            str(point['val']),
            # fontsize=10,
            weight=weight,
        )

# %%

# CVPR PAGE SIZES
# TEXT_WIDTH = 3.25
# PAGE_WIDTH = 6.875
# ECCV PAGE SIZES
TEXT_WIDTH = 4.80
FONTSIZE = 11

def setup_figure(grid=False, fontsize=FONTSIZE):
    # fig_width = TEXT_WIDTH * 1.2
    # fig_height = TEXT_WIDTH
    # Wide version
    fig_width = TEXT_WIDTH * 1.7
    fig_height = TEXT_WIDTH
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams["figure.figsize"] = (fig_width, fig_height)
    plt.rcParams["axes.grid"] = grid
    plt.rcParams.update(
        {
            "font.size": fontsize,
            "legend.fontsize": "medium",
            "axes.labelsize": "large",
            "xtick.labelsize": "medium",
            "ytick.labelsize": "medium",
            "axes.titlesize": "large",
            "lines.linewidth": 1.0,
            "lines.markersize": 7,
        }
    )
    fig = plt.figure(
        figsize=(fig_width, fig_height),
    )
    return fig


# %% - REPLACE MARKER BY HUE CODING
fig = setup_figure(grid=True, fontsize=10)
ax1 = fig.subplots()

# color_palette = sns.color_palette("colorblind")
# sh_cpal = sns.color_palette("Blues")
# sh_cpal = sns.dark_palette("#69d", reverse=True)
# mh_cpal = sns.color_palette("YlOrBr")

hue_dict = {
    "unconstrained": "#D81B60",
    "regularized":   "#FFC107",
    "constrained":   "#004D40",
}




markers_styler = {h: "o" for h in methods["Hypotheses"].unique()}
markers_styler["1"] = "^"
k = 100
markers_sizer = {
    "1": 1*k,
    "3–5": 1*k,
    "10–20": 4*k,
    "200": 9*k,
    }

methods["marker_size"] = methods["Hypotheses"].map(markers_sizer.get)

g1 = sns.scatterplot(
    data=methods,
    x="MPJPE",
    y="MPSCE",
    hue="Constraints",
    size="Hypotheses",
    palette=hue_dict,
    style="Hypotheses",
    markers=markers_styler,
    sizes=markers_sizer,
    zorder=2,
    ax=ax1
)

ax1.xaxis.grid(True, "major", linewidth=.25, alpha=0.75)
ax1.yaxis.grid(True, "major", linewidth=.25, alpha=0.75)
ax1.set_xlabel(
    r"$\it{good}$ $\leftarrow$ Joint Position Error MPJPE (mm) $\rightarrow$ $\it{bad}$",
    # fontsize=11,
)
ax1.set_ylabel(
    r"$\it{good}$ $\leftarrow$ Inconsistency MPSCE (mm) $\rightarrow$ $\it{bad}$",
    # fontsize=11,
)
ax1.set_ylim(-1, 16)
# ax1.set_aspect('equal', adjustable='box')
# ax1.set_title("Human 3.6M", weight="bold")

label_points(
    methods["MPJPE"],
    methods["MPSCE"],
    methods["marker_size"],
    methods["method"],
    ax1,
)

# extract the existing handles and labels
handlers, labels = ax1.get_legend_handles_labels()

# make subtitles italic
labels = [
    ell if ell not in ["Constraints", "Hypotheses"]
    else r"$\it{{{}}}$".format(ell.replace(' ', r'\;'))
    for ell in labels
]
for (h, ell) in zip(handlers, labels):
    if "Hypotheses" in ell:
        break
    if isinstance(h, plt.Line2D):
        h.set_marker("o")


# delete previous legend
# ax1.get_legend().remove()

# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.legend(
    handlers, labels,
    # ncol=1,
    loc="lower left",
    fontsize=9,
    bbox_to_anchor=(1.05, -0.01)
)
fig.tight_layout()


plt.savefig("figures/error_vs_consistency_h36m_and_toy_new_hue.pdf", bbox_inches="tight")
plt.savefig("figures/error_vs_consistency_h36m_and_toy_new_hue.svg", bbox_inches="tight")
plt.savefig("figures/error_vs_consistency_h36m_and_toy_new_hue.png", bbox_inches="tight", dpi=1000)


# %%
