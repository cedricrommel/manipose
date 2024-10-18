# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %%
n_hyps = np.array([2, 3, 5])
ave_mpjpe = np.array([62.4, 56.0, 54.6])
ora_mpjpe = np.array([62.5, 52.2, 52.0])

# %%

beta = np.array([0.1, 0.5, 1.])
ave_mpjpe_per_beta = np.array([53.3, 54.6, 83.6])  # first value uses mup=False ... --> fix
ora_mpjpe_per_beta = np.array([47.4, 52.0, 82.8])  # first value uses mup=False ... --> fix

# %%
# CVPR PAGE SIZES
TEXT_WIDTH = 3.25
PAGE_WIDTH = 6.875
FONTSIZE = 10

def setup_style(grid=False, column_fig=False, fontsize=FONTSIZE):
    # plt.style.use("seaborn-paper")
    plt.style.use("seaborn-v0_8")

    if column_fig:
        plt.rcParams["figure.figsize"] = (TEXT_WIDTH, TEXT_WIDTH / 2)
    else:
        plt.rcParams["figure.figsize"] = (PAGE_WIDTH, PAGE_WIDTH / 2)
    plt.rcParams["axes.grid"] = grid
    # lw = 1.0 if column_fig else 0.5
    plt.rcParams.update(
        {
            "font.size": fontsize,
            "legend.fontsize": "medium",
            "axes.labelsize": "medium",
            "xtick.labelsize": "medium",
            "ytick.labelsize": "medium",
            "axes.titlesize": "medium",
            "lines.linewidth": 1.0,
            "lines.markersize": 7,
        }
    )

# %%
setup_style(grid=True, column_fig=True, fontsize=7)
fig, ax = plt.subplots(
    1, 1,
    figsize=(TEXT_WIDTH, 0.35 * TEXT_WIDTH)
)
c_pal = sns.color_palette("muted")
ax.plot(n_hyps, ave_mpjpe, "--", marker="o", lw=2, label="Aggregated MPJPE", color=c_pal[1])
ax.plot(n_hyps, ora_mpjpe, marker="s", lw=2, label="Oracle MPJPE", color=c_pal[1])
ax.set_xticks([2, 3, 4, 5], [2, 3, 4, 5])
ax.set_xlabel(r"# Hypotheses $K$")
ax.set_ylabel("[mm]")
plt.legend(loc=0)
# plt.tight_layout()

# plt.savefig("figures/nhyps_plots.pdf", bbox_inches="tight")
# %%

setup_style(grid=True, column_fig=True, fontsize=7)
fig, (ax_k, ax_beta) = plt.subplots(
    1, 2,
    figsize=(TEXT_WIDTH, 0.35 * TEXT_WIDTH)
)
c_pal = sns.color_palette("muted")
ax_k.plot(n_hyps, ave_mpjpe, "--", marker="o", lw=2, label="Aggregated MPJPE", color=c_pal[1])
ax_k.plot(n_hyps, ora_mpjpe, marker="s", lw=2, label="Oracle MPJPE", color=c_pal[1])
ax_k.set_xticks([2, 3, 4, 5], [2, 3, 4, 5])
ax_k.set_xlabel(r"# Hypotheses $K$")
ax_k.set_ylabel("[mm]")
# ax_k.legend(loc=0)

ax_beta.plot(beta, ave_mpjpe_per_beta, "--", marker="o", lw=2, label="Aggregated MPJPE", color=c_pal[1])
ax_beta.plot(beta, ora_mpjpe_per_beta, marker="s", lw=2, label="Oracle MPJPE", color=c_pal[1])
ax_beta.set_xticks([0.1, 0.5, 1.])
ax_beta.set_xlabel(r"Score loss weight $\beta$")
ax_beta.set_ylabel("")

handles, labels = ax_beta.get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='upper center',
    ncol=2,
)
fig.tight_layout(rect=[0, 0, 1, 0.85], pad=0.1, h_pad=0.1, w_pad=0.15)

plt.savefig("figures/nhyps_beta_plots.pdf", bbox_inches="tight")
# %%
