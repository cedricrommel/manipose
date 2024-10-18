# %%
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# %% - fetch GT data
data_dir = Path("/home/crommel/shared/crommel/h36m_data")
preproc_dataset_path = data_dir / (
    "preproc_data_3d_h36m_17_mh_so3_hpe.pkl"
)

if preproc_dataset_path.exists():
    print("==> Loading preprocessed dataset...")
    with open(preproc_dataset_path, "rb") as f:
        dataset = pickle.load(f)

# %% - Max stretching

mixste_data_path = Path("/home/crommel/shared/crommel/mlflow_files/mlruns/705717003473657216/24cd074f1df1439abfdf186d74c439fc/artifacts")
manipose_data_path = Path("/home/crommel/shared/crommel/mlflow_files/mlruns/705717003473657216/c6bfd675966c4378baa397d64f806364/artifacts")
# mixste_data_path = Path("/home/crommel/shared/crommel/mlflow_files/mlruns/483843057870309330/2a787541d99b4645906bbf1d6b21424f/artifacts")
# manipose_data_path = Path("/home/crommel/shared/crommel/mlflow_files/mlruns/483843057870309330/c9253820a6d5474da79fe1c62fde441d/artifacts")

# %%

mixste_stretch = pd.read_csv(mixste_data_path / "seg_max_strech.csv", index_col=0)
manipose_stretch = pd.read_csv(manipose_data_path / "seg_max_strech.csv", index_col=0)
mixste_vstretch = pd.read_csv(mixste_data_path / "seg_max_delta_strech.csv", index_col=0)
manipose_vstretch = pd.read_csv(manipose_data_path / "seg_max_delta_strech.csv", index_col=0)

# %%
mixste_ave_stretch = mixste_stretch.loc["average", :].to_frame()
mixste_ave_stretch.columns = ["Average bone stretching [mm]"]
mixste_ave_stretch["model"] = "MixSTE"
mixste_ave_stretch["Segment"] = [f"{j2}-{j1}" for j1, j2 in dataset.skeleton.bones]

manipose_ave_stretch = manipose_stretch.loc["average", :].to_frame()
manipose_ave_stretch.columns = ["Average bone stretching [mm]"]
manipose_ave_stretch["model"] = "ManiPose"
manipose_ave_stretch["Segment"] = [f"{j2}-{j1}" for j1, j2 in dataset.skeleton.bones]

stretching_errs = pd.concat([mixste_ave_stretch, manipose_ave_stretch], axis=0)
stretching_errs = stretching_errs.reset_index(drop=True)

# %%

mixste_ave_vstretch = mixste_vstretch.loc["average", :].to_frame()
mixste_ave_vstretch.columns = ["Max bone stretching speed [mm/frame]"]
mixste_ave_vstretch["model"] = "MixSTE"
mixste_ave_vstretch["Segment"] = [f"{j2}-{j1}" for j1, j2 in dataset.skeleton.bones]

manipose_ave_vstretch = manipose_vstretch.loc["average", :].to_frame()
manipose_ave_vstretch.columns = ["Max bone stretching speed [mm/frame]"]
manipose_ave_vstretch["model"] = "ManiPose"
manipose_ave_vstretch["Segment"] = [f"{j2}-{j1}" for j1, j2 in dataset.skeleton.bones]

vstretching_errs = pd.concat([mixste_ave_vstretch, manipose_ave_vstretch], axis=0)
vstretching_errs = vstretching_errs.reset_index(drop=True)
# %%
sns.barplot(
    data=stretching_errs,
    x="Segment",
    y="Average bone stretching [mm]",
    hue="model",
)
plt.xticks(rotation=90)
# %%
sns.barplot(
    data=vstretching_errs,
    x="Segment",
    y="Max bone stretching speed [mm/frame]",
    hue="model",
)
plt.xticks(rotation=90)

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

# %% -- jointwise errors data
mlflow_base_path = Path("/home/crommel/shared/crommel/mlflow_files/mlruns")
jw_path_mixste = mlflow_base_path / "408976714800957044/477c4759516d4e91bb6370a42704f511/artifacts/jw_err.csv"
jw_path_mhmc = mlflow_base_path / "408976714800957044/ab7c835939564abfb014027b0d8db3ee/artifacts/jw_err.csv"

# %%
jw_mixste = pd.read_csv(jw_path_mixste, index_col=0)
jw_mixste_ave = jw_mixste.loc["average", :].to_frame()
jw_mixste_ave.columns = ["JW-MPJPE [mm]"]
jw_mixste_ave["model"] = "MixSTE"
jw_mixste_ave = jw_mixste_ave.drop(index="Hip")
jw_mixste_ave["Joint"] = list(range(1, 17))

jw_mhmc = pd.read_csv(jw_path_mhmc, index_col=0)
jw_mhmc_ave = jw_mhmc.loc["average", :].to_frame()
jw_mhmc_ave.columns = ["JW-MPJPE [mm]"]
jw_mhmc_ave["model"] = "ManiPose"
# jw_mhmc_ave["model"] = "MHMC"
jw_mhmc_ave = jw_mhmc_ave.drop(index="Hip")
jw_mhmc_ave["Joint"] = list(range(1, 17))

jw_errs = pd.concat([jw_mixste_ave, jw_mhmc_ave], axis=0)
# jw_errs = jw_errs.drop(index="Hip")
# jw_errs["Joint"] = jw_errs.index
jw_errs = jw_errs.reset_index(drop=True)

# %%
setup_style(grid=False, column_fig=False, fontsize=7)
fig, ax_list = plt.subplots(
    2, 1,
    figsize=(TEXT_WIDTH, 1.2 * TEXT_WIDTH)
)
sns.barplot(
    data=jw_errs,
    x="Joint",
    y="JW-MPJPE [mm]",
    hue="model",
    ax=ax_list[0],
)
ax_list[0].set_ylabel("Joint-wise\nMPJPE [mm]")
sns.barplot(
    stretching_errs,
    x="Segment",
    y="Average bone stretching [mm]",
    hue="model",
    ax=ax_list[1],
)
ax_list[1].set_ylabel("Average bone stretching\nper segment [mm]")
for ax in ax_list:
    ax.get_legend().remove()
    # ax.set_xlabel("")
    ax.tick_params(axis='x', labelrotation = 90)
ax_list[0].set_ylim(15, ax_list[0].get_ylim()[1])
handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='upper center',
    ncol=2,
)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("figures/jw_err_and_max_strech.pdf")
# %%
