# %% -- IMPORTS
import pickle
from collections import OrderedDict
from copy import deepcopy
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from data import EasyDist, HardBimodalDist, HardUnimodalDist, LiftingDataset
from matplotlib.patches import Arc
from models import ConstrainedMlp, ConstrainedMlpRmcl, Mlp, SquaredReLU
from omegaconf import OmegaConf
from torch import nn
from torch.nn.functional import mse_loss
from training import Trainer, calc_mpjpe
from utils.plot_utils import (plot_angular_density, plot_circle,
                              plot_predictions)
from utils.utils import polar2cartesian

# %% -- RANDOMNESS
SEED = 42
np_rng = np.random.default_rng(seed=SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# %%

METHODS_NAMES = {
    "mlp": "Unconstr. MLP",
    "constrained": "Constr. MLP",
    "constrained_rmcl": "ManiPose",
    # "constrained_rmcl": "MHMC",
}

reg_cpal = sns.color_palette()
GT_COL = reg_cpal[2]
INP_COL = reg_cpal[5]
sh_cpal = sns.dark_palette("#69d", reverse=True)
MLP_COL = sh_cpal[0]
CONST_COL = sh_cpal[3]
mh_cpal = sns.color_palette("YlOrBr")
MANI_COL = mh_cpal[4]

METHODS_COLORS = {
    "mlp": MLP_COL,
    "constrained": CONST_COL,
    "constrained_rmcl": MANI_COL,
}

# %% CVPR PAGE SIZES
TEXT_WIDTH = 3.25
PAGE_WIDTH = 6.875
FONTSIZE = 10


# %% -- CREATE THREE DISTRIBUTIONS OF INCREASING COMPLEXITY
radius = 1.0  # fixed size for now

easy_distribution = EasyDist(
    radius=radius,
    random_state=np_rng,
)

hard1_distribution = HardUnimodalDist(
    radius=radius,
    random_state=deepcopy(np_rng),
)

hard2_distribution = HardBimodalDist(
    radius=radius,
    random_state=deepcopy(np_rng),
)

# %% - PLOT SETTING


def plot_setting(
    radius=1.0,
    offset=2.0,
    ax=None,
    save_path=None,
    display_legend=True,
    show_angle=True,
):
    ax = plot_circle(
        r=radius,
        show_ticks=False,
        show_input_ax=True,
        show_input_ax_label=True,
        inputs_offset=offset,
        show_output_axs=True,
        ax=ax
    )

    # plot input and output example

    ang_pos_example = np.pi / 3
    x_pos = np.cos(ang_pos_example)
    y_pos = np.sin(ang_pos_example)

    input_J0 = np.array([0., -offset])
    input_J1 = np.array([x_pos, -offset])
    output_J0 = np.array([0., 0.])
    output_J1 = np.array([x_pos, y_pos])
    output = np.stack([output_J0, output_J1], axis=0)

    ax.scatter(*input_J0, marker="o", s=50, color=INP_COL, label="Inputs")
    ax.text(input_J0[0] + 0.1, input_J0[1] + 0.1, s="K0")
    ax.scatter(*input_J1, marker="o", s=50, color=INP_COL)
    ax.text(input_J1[0] + 0.1, input_J1[1] + 0.1, s="K1")
    ax.scatter(*output_J0, marker="o", s=50, color=GT_COL, label="Outputs")
    ax.text(output_J0[0] - 0.2, output_J0[1] + 0.1, s="J0")
    ax.scatter(*output_J1, marker="o", s=50, color=GT_COL)
    ax.text(output_J1[0], output_J1[1] + 0.2, s="J1")

    # connect outputs joints with segment
    ax.text(output[:, 0].sum() / 2 - 0.15, output[:, 1].sum() / 2, s=r"$s$")
    ax.plot(output[:, 0], output[:, 1], "k-", lw=2, zorder=0)

    # show inputs and outputs
    ax.vlines(
        x=[0., x_pos],
        ymin=[-offset] * 2,
        ymax=[0., y_pos],
        ls="--",
        lw=1.5,
        color="grey",
        zorder=0
    )
    if show_angle:
        angle = Arc(
            xy=(0., 0.),
            width=0.5 * radius,
            height=0.5 * radius,
            angle=0.,
            theta1=0.,
            theta2=ang_pos_example * 180 / np.pi,
            color="grey",
            linewidth=1.5
        )
        ax.add_patch(angle) # To display the angle arc
        ax.text(0.3 * radius, 0.1 * radius, r"$\theta$")

    if display_legend:
        ax.legend(loc="lower left")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")


# plot_setting(
#     # save_path="./figures/toy-exp-setting.pdf",
# )


# %%
def plot_dist_and_samples(
    distribution,
    X,
    Y,
    offset,
    ax=None,
    display_legend=True,
    omit_targets=False,
):

    t = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    pdf = distribution.pdf(t)

    # Use the two plots together
    # fig = plt.figure(figsize=(8, 6))
    # fig = plt.figure(figsize=(15, 12))
    ax = plot_angular_density(
        t, pdf * 0.5,
        colour=GT_COL,
        # colour="green",
        show_ticks=False,
        show_center=True,
        label="GT probability",
        # label="prob. density",
        ax=ax,
        inputs_offset=offset,
    )

    if not omit_targets:
        ax.scatter(
            Y[:, 0],
            Y[:, 1],
            marker="o",
            label="Outputs",
            c=GT_COL
            # c=palette[1],
            # alpha=0.5
        )
    ax.scatter(
        X,
        np.ones_like(X) * -offset,
        label="Inputs",
        # alpha=0.5,
        c=INP_COL,
        # c=palette[0],
    )
    if display_legend:
        ax.legend(loc="upper left")
    return plt.gcf(), ax


# %% - UPDATE SAMPLE PLOTTING FUNC

from logging import warning
from pathlib import Path
from typing import Dict, Union

import mlflow as mlf
from data import LiftingDist1Dto2D
from matplotlib.colors import ListedColormap


def save_fig_and_log(fig, save_path, log_to_mlf):
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    if log_to_mlf:
        if save_path is None:
            warning(
                "Cannot log to MLFlow unless the figure is saved somewhere."
                "Try again with save_path not equal to None.",
                UserWarning,
            )
        else:
            mlf.log_artifact(save_path)


def plot_predictions(
    distribution: LiftingDist1Dto2D,
    X_test: np.array,
    Y_test: np.array,
    predictions_dict: Dict[str, np.array],
    offset: float = 2.0,
    # palette: ListedColormap = sns.color_palette("muted"),
    save_path: Union[str, Path] = None,
    log_to_mlf: bool = False,
    ax: Optional[plt.Axes] = None,
    display_legend: bool = True,
    omit_targets: bool = False,
):
    fig, ax = plot_dist_and_samples(
        distribution, X_test, Y_test, offset, ax=ax,
        display_legend=display_legend,
        omit_targets=omit_targets,
    )

    # plot predictions
    for i, (method, predictions) in enumerate(predictions_dict.items()):
        method_name = METHODS_NAMES[method]
        if method == "constrained_rmcl":
            method_name += " - Aggr."
        ax.scatter(
            predictions[:, 0],
            predictions[:, 1],
            marker="X",
            label=method_name,
            c=METHODS_COLORS[method],
            # c=palette[3 + i],
            # alpha=0.6,
        )
    if display_legend:
        plt.legend(loc="upper left")

    plt.tight_layout()

    save_fig_and_log(fig, save_path, log_to_mlf)

# %% - FUNCTION FOR TRAINING AND PLOTTING PREDICTIONS


def train_and_plot_preds(
    distribution,
    datasets,
    train_loader,
    models_names,
    tr_configs,
    preds_per_model=None,
    hyps_per_model=None,
    trained_models=None,
    save_path=None,
    device="cuda",
    seed=SEED,
    cfg=None,
    ax=None,
):
    if cfg is None:
        cfg = OmegaConf.load("./conf/config.yaml")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # pick network activation function
    if cfg.model.act == "relu":
        act = nn.ReLU
    elif cfg.model.act == "tanh":
        act = nn.Tanh
    elif cfg.model.act == "sqrelu":
        act = SquaredReLU
    else:
        raise ValueError(
            "Currently supported activations are 'relu' and 'tanh'."
            f"Got {cfg.model.act}."
        )

    # create optimizer and trainer
    if cfg.train.optim == "adam":
        optim_cls = torch.optim.Adam
    elif cfg.train.optim == "sgd":
        optim_cls = torch.optim.SGD
    else:
        raise ValueError(
            "Currently supported optim_cls values are 'adam' and 'sgd'."
            f"Got {optim_cls}."
        )

    # create lr scheduler
    if cfg.train.lr_scheduler:
        sched_cls = torch.optim.lr_scheduler.ReduceLROnPlateau
    else:
        sched_cls = None

    if (
        preds_per_model is None or
        hyps_per_model is None or
        trained_models is None
    ):
        preds_per_model = {}
        trained_models = {}
        hyps_per_model = {}

    for model_name, tr_config in zip(models_names, tr_configs):
        if (
            model_name not in preds_per_model or
            model_name not in hyps_per_model or
            model_name not in trained_models
        ):
            cfg.train.update(tr_config)

            # create model
            if model_name == "mlp":
                model = Mlp(
                    in_features=1,
                    hidden_features=cfg.model.hidden_features,
                    out_features=2,
                    n_layers=cfg.model.layers,
                    act_layer=act,
                )
            elif model_name == "constrained":
                model = ConstrainedMlp(
                    in_features=1,
                    hidden_features=cfg.model.hidden_features,
                    out_features=1,
                    n_layers=cfg.model.layers,
                    act_layer=act,
                    radius=cfg.data.radius,
                )
            elif model_name == "constrained_rmcl":
                model = ConstrainedMlpRmcl(
                    in_features=1,
                    hidden_features=cfg.model.hidden_features,
                    out_features=1,
                    n_layers=cfg.model.layers,
                    act_layer=act,
                    radius=cfg.data.radius,
                    n_hyp=cfg.multi_hyp.nsamples,
                    beta=cfg.model.beta,
                )
            else:
                raise ValueError(
                    "Possible 'arch' values are 'mlp' and 'constrained'."
                    f"Got {model_name}."
                )

            trainer = Trainer(
                model=model,
                optim_cls=optim_cls,
                sched_cls=sched_cls,
                checkpointing_dir="figures_data",
                lr=cfg.train.lr,
                config_train=cfg.train,
                device=device,
            )

            trainer.train(
                epochs=cfg.train.epochs,
                loader=train_loader,
                loss_func=mse_loss,
                val_data=datasets.validation_set,
                log_in_mlf=False
            )

            _, (test_predictions,), pred_hypothesis = trainer.eval(
                eval_sets=(datasets.test_set,),
                metric=calc_mpjpe,
            )

            preds_per_model[model_name] = test_predictions.cpu().numpy()
            trained_models[model_name] = deepcopy(trainer.model)
            hyps_per_model[model_name] = pred_hypothesis

    plot_predictions(
        distribution=distribution,
        X_test=datasets.X_test.cpu().numpy(),
        Y_test=datasets.Y_test.cpu().numpy(),
        predictions_dict=preds_per_model,
        save_path=save_path,
        log_to_mlf=False,
        ax=ax,
    )

    palette = sns.color_palette("muted")
    hyp_markers = ["s", "^", "v", "D", "P"]
    for i, (model_name, pred_hypothesis) in enumerate(hyps_per_model.items()):
        if pred_hypothesis is not None:
            pred_hypothesis = pred_hypothesis[0].cpu().numpy()
            for hyp_idx in range(pred_hypothesis.shape[1]):
                hyp_x = pred_hypothesis[:, hyp_idx, 0]
                hyp_y = pred_hypothesis[:, hyp_idx, 1]
                score = pred_hypothesis[:, hyp_idx, 2]

                plt.scatter(
                    hyp_x,
                    hyp_y,
                    marker=hyp_markers[hyp_idx],
                    label=f"{METHODS_NAMES[model_name]} - Hyp. {hyp_idx}",
                    # label=f"{model_name} - h{hyp_idx}",
                    c=palette[3 + i],
                    alpha=0.4,
                )
                plt.plot(
                    [hyp_x, (1 + score) * hyp_x],
                    [hyp_y, (1 + score) * hyp_y],
                    c=palette[3 + i],
                    ls="--",
                    alpha=0.4,
                    # label="scores" if hyp_idx == 0 else None,
                )

    # fix legend and layout and overwrite previous fig
    plt.legend(loc="upper left")
    plt.tight_layout()

    save_fig_and_log(plt.gcf(), save_path=save_path, log_to_mlf=False)

    return preds_per_model, trained_models, hyps_per_model

# %% - FUNCTION TO PLOT ORACLES AND PREDICTIONS


def plot_oracle_and_pred(
    distribution,
    query,
    accept_outputs,
    acc_outputs_probs,
    euclidean_oracle,
    riemanian_oracle,
    models,
    inputs_offset=2.0,
    save_path=None,
    display_legend=True,
    ax=None,
):
    # convert np query to torch
    if isinstance(query, np.ndarray):
        query = torch.from_numpy(query).float().to("cuda").repeat((10, 1))

    # create fictitious batch for forward when necessary
    if len(query.shape) == 1:
        query = query[:, None]
    if sum(query.shape) == 1:
        query = query.repeat((10, 1))

    # forward pass over models
    with torch.no_grad():
        predictions_dict = {
            model_name: model(query)[:1, :].cpu().numpy()
            for model_name, model in models.items()
        }

    # convert query back to numpy
    query = query.cpu().numpy()[0]

    # palette = sns.color_palette("muted")

    # plot the angular density
    t = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    pdf = distribution.pdf(t)
    ax = plot_angular_density(
        t, pdf * 0.5,
        colour=GT_COL,
        # colour="green",
        show_ticks=False,
        show_center=True,
        label="GT probability",
        ax=ax,
        inputs_offset=inputs_offset,
    )

    # plot the query
    ax.scatter(
        query,
        -inputs_offset,
        label="Inputs",
        c=INP_COL,
        # c=palette[0],
    )

    # plot acceptable outputs
    ax.scatter(
        accept_outputs[:, 0],
        accept_outputs[:, 1],
        marker="*",
        label="Constr. MH min.",
        # label="Acceptable outputs",
        c=MANI_COL,
        # c="green",
        s=100,
    )

    # plot corresponding probabilities of each acceptable output
    for p, out in zip(acc_outputs_probs, accept_outputs):
        # ax.plot(
        #     [out[0], (1 + p) * out[0]],
        #     [out[1], (1 + p) * out[1]],
        #     color="green",
        #     ls="--",
        # )
        ax.text(
            x=out[0] * 1.1,
            y=out[1] * 1.1,
            s=f"$p(y|x)={p:.2f}$",
            color=MANI_COL,
            # color="green",
        )

    # plot euclidean oracle, together with explicative line
    ax.scatter(
        euclidean_oracle[:, 0],
        euclidean_oracle[:, 1],
        marker="o",
        label="MSE minimizer",
        c=MLP_COL
        # c=palette[3],
    )
    ax.vlines(
        query, -inputs_offset, np.sin(distribution.modes[0]),
        linestyle="--", color='grey', lw=1.5, zorder=0
    )

    # plot Riemannian oracle
    ax.scatter(
        riemanian_oracle[:, 0],
        riemanian_oracle[:, 1],
        marker="o",
        label="Constr. MSE min.",
        # label="Constr. MSE min.",
        c=CONST_COL,
        # c=palette[4],
    )

    # plot predictions
    for i, (method, predictions) in enumerate(predictions_dict.items()):
        ax.scatter(
            predictions[:, 0],
            predictions[:, 1],
            marker="X",
            label=METHODS_NAMES[method],
            c=METHODS_COLORS[method],
            # c=palette[3 + i],
        )

    if display_legend:
        ax.legend(loc="lower left")
    plt.tight_layout()

    save_fig_and_log(plt.gcf(), save_path=save_path, log_to_mlf=False)
    return ax

# %% - SAMPLE FROM EASY DIST
n_train = 1000
n_val = 1000
n_test = 4
# n_test = 5
# n_test = 10
batch_size = 100
workers = 2

all_datasets = LiftingDataset(
    distribution=easy_distribution,
    n_train=n_train,
    n_val=n_val,
    n_test=n_test,
)
train_loader = all_datasets.get_tr_loader(
    batch_size=batch_size,
    num_workers=workers,
)

X, Y = all_datasets.test_set.tensors

# %% - TRAIN SIMPLE MLP
mlp_tr_conf = OmegaConf.load("./conf/train/mlp_easy.yaml")

preds_per_model_e, trained_models_e, hyps_per_model_e = train_and_plot_preds(
    distribution=easy_distribution,
    datasets=all_datasets,
    train_loader=train_loader,
    models_names=["mlp"],
    tr_configs=[mlp_tr_conf],
    save_path="./figures/easy_mlp_preds.pdf",
)

# %% - SAME FOR UNIMOD HARD DISTRIBUTION

hard1_datasets = LiftingDataset(
    distribution=hard1_distribution,
    n_train=n_train,
    n_val=n_val,
    n_test=n_test,
)
hard1_loader = hard1_datasets.get_tr_loader(
    batch_size=batch_size,
    num_workers=workers,
)

# X, Y = hard1_datasets.test_set.tensors

# %%

mlp_h1_tr_conf = OmegaConf.load("./conf/train/mlp_hard1.yaml")
mlp_h1_tr_conf.epochs = 10
mlp_h1_tr_conf.lr = 0.12

preds_per_model_h1, trained_models_h1, hyps_per_model_h1 = train_and_plot_preds(
    distribution=hard1_distribution,
    datasets=hard1_datasets,
    train_loader=hard1_loader,
    models_names=["mlp"],
    tr_configs=[mlp_h1_tr_conf],
    save_path="./figures/hard1_mlp_preds.pdf",
)

# %% - SAME FOR BIMODAL HARD DISTRIBUTION

hard2_datasets = LiftingDataset(
    distribution=hard2_distribution,
    n_train=n_train,
    n_val=n_val,
    n_test=n_test,
)
hard2_loader = hard2_datasets.get_tr_loader(
    batch_size=batch_size,
    num_workers=workers,
)

# X, Y = hard1_datasets.test_set.tensors

# %%

mlp_h2_tr_conf = OmegaConf.load("./conf/train/mlp_hard2.yaml")

preds_per_model_h2, trained_models_h2, hyps_per_model_h2 = train_and_plot_preds(
    distribution=hard2_distribution,
    datasets=hard2_datasets,
    train_loader=hard2_loader,
    models_names=["mlp"],
    tr_configs=[mlp_h2_tr_conf],
    save_path="./figures/hard2_mlp_preds.pdf",
)

# %% - UNIMODAL HARD WITH FREE AND CONSTRAINED MLP

const_h1_tr_conf = OmegaConf.load("./conf/train/constrained_hard1.yaml")

preds_per_model_h1, trained_models_h1, hyps_per_model_h1 = train_and_plot_preds(
    distribution=hard1_distribution,
    datasets=hard1_datasets,
    train_loader=hard1_loader,
    models_names=["mlp", "constrained"],
    tr_configs=[mlp_h1_tr_conf, const_h1_tr_conf],
    preds_per_model=preds_per_model_h1,
    trained_models=trained_models_h1,
    hyps_per_model=hyps_per_model_h1,
    save_path="./figures/hard1_mlp_const_preds.pdf",
)

# %% - BIMODAL HARD WITH FREE AND CONSTRAINED MLP

const_h2_tr_conf = OmegaConf.load("./conf/train/constrained_hard2.yaml")

preds_per_model_h2, trained_models_h2, hyps_per_model_h2 = train_and_plot_preds(
    distribution=hard2_distribution,
    datasets=hard2_datasets,
    train_loader=hard2_loader,
    models_names=["mlp", "constrained"],
    tr_configs=[mlp_h2_tr_conf, const_h2_tr_conf],
    save_path="./figures/hard2_mlp_const_preds.pdf",
    preds_per_model=preds_per_model_h2,
    trained_models=trained_models_h2,
    hyps_per_model=hyps_per_model_h2,
)

# %% - PLOTTING ORACLES AND PREDICTIONS FOR BIMODAL DIST

# creating special query for which the oracle outputs are easy to compute

query_input = np.unique(np.cos(hard2_distribution.modes), )[None, :]

euclidean_oracle_height = np.sum(
    np.sin(hard2_distribution.modes) * hard2_distribution.weights,
    keepdims=True,
)[:, None]

acceptable_outputs = np.hstack([
    np.cos(hard2_distribution.modes)[:, None],
    np.sin(hard2_distribution.modes)[:, None]
])

acceptable_outputs_probs = hard2_distribution.weights

euclidean_oracle_output = np.hstack([query_input, euclidean_oracle_height])

angular_oracle = np.sum(
    hard2_distribution.modes * hard2_distribution.weights
)

manifold_oracle_output = np.array(
    polar2cartesian(r=radius, theta=angular_oracle),
)[None, :]

# %%
if "constrained_rmcl" in trained_models_h2:
    trained_models_h2_no_rmcl = {
        k: v for k, v in trained_models_h2.items() if k != "constrained_rmcl"
    }
else:
    trained_models_h2_no_rmcl = trained_models_h2

plot_oracle_and_pred(
    distribution=hard2_distribution,
    query=query_input,
    accept_outputs=acceptable_outputs,
    acc_outputs_probs=acceptable_outputs_probs,
    euclidean_oracle=euclidean_oracle_output,
    riemanian_oracle=manifold_oracle_output,
    models=trained_models_h2_no_rmcl,
    inputs_offset=1.5,
    save_path="./figures/oracles_and_preds_regression.pdf",
)

# plot_oracle_and_pred(
#     distribution=hard2_distribution,
#     query=query_input,
#     euclidean_oracle=euclidean_oracle_output,
#     riemanian_oracle=manifold_oracle_output,
#     models=trained_models,
#     inputs_offset=offset,
#     # save_path="./figures/oracles_and_preds_regression.pdf",
# )

# %%

rmcl_h2_tr_conf = OmegaConf.load("./conf/train/rmcl_constrained_hard2.yaml")
cfg = OmegaConf.load("./conf/config.yaml")
cfg.multi_hyp.nsamples = 2

preds_per_model_h2, trained_models_h2, hyps_per_model_h2 = train_and_plot_preds(
    distribution=hard2_distribution,
    datasets=hard2_datasets,
    train_loader=hard2_loader,
    models_names=["mlp", "constrained", "constrained_rmcl"],
    tr_configs=[mlp_h2_tr_conf, const_h2_tr_conf, rmcl_h2_tr_conf],
    # save_path="./figures/hard2_mlp_const_rmcl_preds.pdf",
    save_path="./figures/hard2_mlp_const_rmcl_preds_with_scores.pdf",
    cfg=cfg,
    preds_per_model=preds_per_model_h2,
    trained_models=trained_models_h2,
    hyps_per_model=hyps_per_model_h2,
)

# %%
ax = plot_oracle_and_pred(
    distribution=hard2_distribution,
    query=query_input,
    euclidean_oracle=euclidean_oracle_output,
    riemanian_oracle=manifold_oracle_output,
    models={
        name: model for name, model in trained_models_h2.items()
        if name != "constrained_rmcl"
    },
    inputs_offset=1.5,
    # save_path="./figures/oracles_and_preds_regression_2.pdf",
    save_path=None,
)


rmcl_model = trained_models_h2["constrained_rmcl"]
fake_query_batch = torch.from_numpy(
    query_input).float().to("cuda").repeat((10, 1))
with torch.no_grad():
    rmcl_hyps = rmcl_model(fake_query_batch)[:1, :].cpu().numpy()


with torch.no_grad():
    rmcl_agg_pred = rmcl_model.aggregate(
        rmcl_model(fake_query_batch)
    )[:1, :].cpu().numpy()


# palette = sns.color_palette("muted")
# for hyp_idx in range(rmcl_hyps.shape[1]):
#     ax.scatter(
#         rmcl_hyps[:, hyp_idx, 0],
#         rmcl_hyps[:, hyp_idx, 1],
#         marker="s",
#         label=f"rMCL - hyp {hyp_idx + 1}",
#         c=palette[5],
#     )

# ax.scatter(
#     rmcl_agg_pred[:, 0],
#     rmcl_agg_pred[:, 1],
#     marker="x",
#     label="rMCL - aggreg.",
#     c=palette[6],
# )

# plt.legend(loc="lower left")
# plt.tight_layout()

# save_fig_and_log(
#     plt.gcf(),
#     save_path="./figures/oracles_and_preds_regression_2.pdf",
#     log_to_mlf=False
# )

# %%
data_dict = {
    "A)": {
        "distribution": easy_distribution,
        "dataset": all_datasets,
        "preds_per_model": preds_per_model_e,
        "hyps_per_model": hyps_per_model_e,
        "trained_models": trained_models_e,
    },
    "B)": {
        "distribution": hard1_distribution,
        "dataset": hard1_datasets,
        "preds_per_model": preds_per_model_h1,
        "hyps_per_model": hyps_per_model_h1,
        "trained_models": trained_models_h1,
    },
    "C)": {
        "distribution": hard2_distribution,
        "dataset": hard2_datasets,
        "preds_per_model": preds_per_model_h2,
        "hyps_per_model": hyps_per_model_h2,
        "trained_models": trained_models_h2,
    },
}

# %% SAVE PREDICTIONS FOR LATTER USE
# with open("./figures_data/plot_data_dict_4samples.pkl", "wb") as f:
#     pickle.dump(data_dict, f)

# %% LOAD PREDICTIONS
with open("./figures_data/plot_data_dict_4samples.pkl", "rb") as f:
    data_dict = pickle.load(f)

# %% PLOT PREDICTIONS ON ALL SCENARIOS IN SAME FIGURE


def plot_hyps(hyps_per_model, ax):
    palette = sns.color_palette("muted")
    hyp_markers = ["s", "^", "v", "D", "P"]
    for i, (model_name, pred_hypothesis) in enumerate(hyps_per_model.items()):
        if pred_hypothesis is not None:
            pred_hypothesis = pred_hypothesis[0].cpu().numpy()
            for hyp_idx in range(pred_hypothesis.shape[1]):
                hyp_x = pred_hypothesis[:, hyp_idx, 0]
                hyp_y = pred_hypothesis[:, hyp_idx, 1]
                score = pred_hypothesis[:, hyp_idx, 2]

                ax.scatter(
                    hyp_x,
                    hyp_y,
                    marker=hyp_markers[hyp_idx],
                    label=f"{METHODS_NAMES[model_name]} - Hyp. {hyp_idx}",
                    # label=f"{model_name} - h{hyp_idx}",
                    c=METHODS_COLORS[model_name],
                    # c=palette[3 + i],
                    alpha=0.6,
                )
                ax.plot(
                    [hyp_x, (1 + score) * hyp_x],
                    [hyp_y, (1 + score) * hyp_y],
                    # c=palette[3 + i],
                    c=METHODS_COLORS[model_name],
                    ls="--",
                    alpha=0.6,
                    lw=2.,
                    label=(
                        f"{METHODS_NAMES[model_name]} - " + r"scores $\gamma_k$"
                        if hyp_idx > 0 else None
                    ),
                )


def setup_style(grid=False, column_fig=False, fontsize=FONTSIZE):
    plt.style.use("seaborn-paper")

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


def group_plot(data_dict, save_path):
    setup_style(grid=False, column_fig=False, fontsize=11)

    # Create axes
    fig, ax_list = plt.subplots(
        1, len(data_dict),
        sharex=True,
        sharey=True,
        figsize=(PAGE_WIDTH, PAGE_WIDTH / 2),
    )

    # Just plot
    for i, (scenario, data) in enumerate(data_dict.items()):
        ax = ax_list[i]
        ax.set_title(scenario)
        plot_predictions(
            distribution=data["distribution"],
            X_test=data["dataset"].X_test.cpu().numpy(),
            Y_test=data["dataset"].Y_test.cpu().numpy(),
            predictions_dict=data["preds_per_model"],
            save_path=None,
            log_to_mlf=False,
            ax=ax,
            display_legend=False,
            omit_targets=True,
            offset=1.5,
        )

        plot_hyps(data["hyps_per_model"], ax)

    # Handle legends
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))  # removes duplicate legends
    fig.legend(
        by_label.values(), by_label.keys(),
        loc='lower center',
        ncol=3,
    )
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")


# %%

# save_path = "./figures/predictions_all_scenarios_4samples_manipose.pdf"
# save_path = "./figures/predictions_all_scenarios_4samples.pdf"
# group_plot(data_dict, save_path)

# %% PLOT SETTING AND ORACLE ON SAME FIGURE

def create_oracle_minimizers(data_dict):
    trained_models_h2 = deepcopy(data_dict["C)"]["trained_models"])
    hard2_distribution = deepcopy(data_dict["C)"]["distribution"])
    if "constrained_rmcl" in trained_models_h2:
        trained_models_h2_no_rmcl = {
            k: v for k, v in trained_models_h2.items()
            if k != "constrained_rmcl"
        }
    else:
        trained_models_h2_no_rmcl = trained_models_h2

    query_input = np.unique(np.cos(hard2_distribution.modes), )[None, :]

    euclidean_oracle_height = np.sum(
        np.sin(hard2_distribution.modes) * hard2_distribution.weights,
        keepdims=True,
    )[:, None]

    acceptable_outputs = np.hstack([
        np.cos(hard2_distribution.modes)[:, None],
        np.sin(hard2_distribution.modes)[:, None]
    ])

    acceptable_outputs_probs = hard2_distribution.weights

    euclidean_oracle_output = np.hstack([query_input, euclidean_oracle_height])

    angular_oracle = np.sum(
        hard2_distribution.modes * hard2_distribution.weights
    )

    manifold_oracle_output = np.array(
        polar2cartesian(r=1.0, theta=angular_oracle),
    )[None, :]

    return (
        query_input,
        trained_models_h2_no_rmcl,
        acceptable_outputs,
        acceptable_outputs_probs,
        euclidean_oracle_output,
        manifold_oracle_output,
    )


def group_plot_setting(data_dict, save_path):
    setup_style(grid=False, column_fig=False)

    # Create axes
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        sharex=True,
        sharey=True,
        # figsize=(TEXT_WIDTH, TEXT_WIDTH / 2),
        figsize=(PAGE_WIDTH, PAGE_WIDTH / 2),
    )

    # Plot setting
    plot_setting(
        ax=ax1,
        display_legend=False,
        offset=1.5,
    )

    # Plot oracles
    (
        query_input,
        trained_models_h2_no_rmcl,
        acceptable_outputs,
        acceptable_outputs_probs,
        euclidean_oracle_output,
        manifold_oracle_output,
    ) = create_oracle_minimizers(data_dict)

    plot_oracle_and_pred(
        distribution=hard2_distribution,
        query=query_input,
        inputs_offset=1.5,
        accept_outputs=acceptable_outputs,
        acc_outputs_probs=acceptable_outputs_probs,
        euclidean_oracle=euclidean_oracle_output,
        riemanian_oracle=manifold_oracle_output,
        models=trained_models_h2_no_rmcl,
        display_legend=False,
        ax=ax2,
    )

    # Handle legends
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles2.pop(1)  # remove duplicate input
    labels2.pop(1)
    handles += handles2
    labels += labels2
    by_label = OrderedDict(zip(labels, handles))  # removes duplicate legends
    fig.legend(
        by_label.values(), by_label.keys(),
        loc='lower center',
        ncol=3,
    )
    fig.tight_layout(rect=[0, 0.2, 1, 1])

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

# %%

# save_path = "./figures/setting_and_oracles_corrected2.pdf"
# # save_path = "./figures/setting_and_oracles.pdf"
# group_plot_setting(data_dict, save_path=save_path)
# %%

def plot_single_figure(data_dict, save_path, col=True):
    setup_style(grid=False, column_fig=False)

    # Create axes
    if col:
        fig, ax_arr = plt.subplots(
            2, 2,
            sharex=True,
            sharey=True,
            # figsize=(TEXT_WIDTH, TEXT_WIDTH / 2),
            figsize=(PAGE_WIDTH, PAGE_WIDTH),
        )
        ax1 = ax_arr[0, 0]
        ax2 = ax_arr[0, 1]
        ax3 = ax_arr[1, 0]
        ax4 = ax_arr[1, 1]
        ax_list = [ax1, ax2, ax3, ax4]
    else:
        fig, ax_list = plt.subplots(
            1, 4,
            sharex=True,
            sharey=True,
            # figsize=(TEXT_WIDTH, TEXT_WIDTH / 2),
            figsize=(2 * PAGE_WIDTH, PAGE_WIDTH / 2),
        )
        ax1, ax2, ax3, ax4 = ax_list

    # 

    # Plot setting
    plot_setting(
        ax=ax1,
        display_legend=False,
        offset=1.5,
    )
    ax1.set_title("(A)", weight="bold")

    # Plot oracles
    (
        query_input,
        trained_models_h2_no_rmcl,
        acceptable_outputs,
        acceptable_outputs_probs,
        euclidean_oracle_output,
        manifold_oracle_output,
    ) = create_oracle_minimizers(data_dict)

    plot_oracle_and_pred(
        distribution=hard2_distribution,
        query=query_input,
        inputs_offset=1.5,
        accept_outputs=acceptable_outputs,
        acc_outputs_probs=acceptable_outputs_probs,
        euclidean_oracle=euclidean_oracle_output,
        riemanian_oracle=manifold_oracle_output,
        models=trained_models_h2_no_rmcl,
        display_legend=False,
        ax=ax2,
    )
    ax2.set_title("(B)", weight="bold")

    # Remove scenario B and rename C
    copied_data_dict = data_dict.copy()
    copied_data_dict.pop("B)")

    # Remove aggregated ManiPose predictions
    copied_data_dict["C)"]["preds_per_model"] = {
        k: v
        for k, v in copied_data_dict["C)"]["preds_per_model"].items()
        if k != "constrained_rmcl"
    }

    titles = ["(C)", "(D)"]
    pred_axis = [ax3, ax4]

    # Just plot
    for i, data in enumerate(copied_data_dict.values()):
        ax = pred_axis[i]
        # ax.set_title(scenario)
        ax.set_title(titles[i], weight="bold")
        plot_predictions(
            distribution=data["distribution"],
            X_test=data["dataset"].X_test.cpu().numpy(),
            Y_test=data["dataset"].Y_test.cpu().numpy(),
            predictions_dict=data["preds_per_model"],
            save_path=None,
            log_to_mlf=False,
            ax=ax,
            display_legend=False,
            omit_targets=True,
            offset=1.5,
        )

        plot_hyps(data["hyps_per_model"], ax)

    # Handle legends
    handles = []
    labels = []
    for ax in ax_list:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l

    # removes duplicate legends
    desired_order = [
        "Inputs",
        "Outputs",
        "GT probability",
        "MSE minimizer",
        "Constr. MSE min.",
        "Constr. MH min.",
        "Unconstr. MLP",
        "Constr. MLP",
        "ManiPose - Hyp. 0",
        "ManiPose - Hyp. 1",
        r"ManiPose - scores $\gamma_k$",
    ]
    unord_labels = dict(zip(labels, handles))
    by_label = OrderedDict({k: unord_labels[k] for k in desired_order})
    n_col = 3 if col else 6    
    fig.legend(
        by_label.values(), by_label.keys(),
        loc='lower center',
        ncol=n_col,
    )
    fig.tight_layout(rect=[0, 0.1, 1, 1])

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

# %%
# I need to:
# x make 2,2 figure, with shared legend
# x remove scenario B
# x remove aggregated ManiPose
# x improve legend: Query input, Unconst. MLP, Constr. MLP, Constr. MSE minimizer, Constr. MH minimizer
# x legend order
# x align colors with rest of paper
# x remove pi
# x add theta
# x adapt to 1,4 format as well

path_2_2 = "./figures/single_toy_picture_column.pdf"
plot_single_figure(data_dict, path_2_2)
path_2_2 = "./figures/single_toy_picture_column.png"
plot_single_figure(data_dict, path_2_2)
# %%
