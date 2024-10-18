import os
from pathlib import Path

from logging import warning
import hydra
import mlflow as mlf
import numpy as np
import torch
import torch.nn.functional as F
from data import (EasyDist, HardBimodalDist, HardQuadmodalDist,
                  HardUnimodalDist, LiftingDataset, LiftingDatasetV2, LiftingDist2Dto3D)
from models import (ConstrainedMlp, ConstrainedMlpRmcl, ConstrainedMlpV2, ConstrainedMlpRmclV2, LiftingDiffusionModel,
                    Mlp, SquaredReLU)
from omegaconf import DictConfig, OmegaConf
from torch import nn
from training import Trainer, calc_mpjpe, distance_to_circle, calc_mpjpe_3D, std_length
from utils.plot_utils import plot_predictions, plot_training_curve
from utils.utils import (log_params_from_omegaconf_dict,
                         save_and_log_np_artifact, set_random_seeds)
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", project_root_env_var=True, pythonpath=True)
# The above line allows the script to find the project root directory, and to set the PROJECT_ROOT environment variable

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("==> Using settings:")
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: %s" % device)

    cwd = Path(os.getcwd())  # hydra run dir
    output_dir = cwd / cfg.run.experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    # ckpts_dir = Path(os.getcwd()) / cfg.run.output_dir
    # ckpts_dir.mkdir(exist_ok=True)

    # Set seeds for init reproducibility
    print(f"==> Setting seeds to {cfg.run.seed} for init")
    set_random_seeds(
        seed=cfg.run.seed,
        cuda=True,
        cudnn_benchmark=cfg.run.cudnn_benchmark,
        set_deterministic=cfg.run.set_deterministic,
    )

    # create data distribution in correct scenario
    if cfg.data.scenario == "easy":
        distribution = EasyDist(
            radius=cfg.data.radius,
            random_state=cfg.run.seed,
        )
    elif cfg.data.scenario == "hard-1":
        distribution = HardUnimodalDist(
            radius=cfg.data.radius,
            random_state=cfg.run.seed,
        )
    elif cfg.data.scenario == "hard-2":
        distribution = HardBimodalDist(
            radius=cfg.data.radius,
            random_state=cfg.run.seed,
        )
    elif cfg.data.scenario == "hard-4":
        distribution = HardQuadmodalDist(
            radius=cfg.data.radius,
            random_state=cfg.run.seed,
        )
    elif cfg.data.scenario == "torus-2Dto3D":
        distribution = LiftingDist2Dto3D(
            major_radius=cfg.data.major_radius,
            minor_radius=cfg.data.minor_radius,
            weights= [0.3, 0.4,0.2,0.1],
            modes = [(-3.1415, 0), (0,3.1415/4),(0.5, -3.1415/4), (2*3.1415/3,3.1415/2)],
            dispersions=[(2,2), (4,4),(3,3),(10,10)],
            random_state=cfg.run.seed)
    else:
        raise ValueError(
            "Possible values for scenario are 'easy', 'hard-1', 'hard-2', 'hard-4' or "
            f"'torus-2Dto3D'. Got {cfg.data.scenario}."
        )
    
    if '3D' in cfg.data.scenario :
        all_datasets = LiftingDatasetV2(
            distribution=distribution,
            n_train=cfg.data.n_train,
            n_val=cfg.data.n_val,
            n_test=cfg.data.n_test,
        )

    else:
        # sample data and create dataloaders
        all_datasets = LiftingDataset(
            distribution=distribution,
            n_train=cfg.data.n_train,
            n_val=cfg.data.n_val,
            n_test=cfg.data.n_test,
        )
    train_loader = all_datasets.get_tr_loader(
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.workers,
    )

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

    # create model
    if cfg.diffusion.enabled:
        model = LiftingDiffusionModel(
            config=cfg,
            act=act,
            device=device,
        )
    elif '3D' in cfg.data.scenario :
        if cfg.model.arch == "mlp":
            model = Mlp(
                in_features=2,
                hidden_features=cfg.model.hidden_features,
                out_features=6,
                n_layers=cfg.model.layers,
                act_layer=act,
            )
        elif cfg.model.arch == "constrained":
            model = ConstrainedMlpV2(
                in_features=2,
                hidden_features=cfg.model.hidden_features,
                out_features=2,
                n_layers=cfg.model.layers,
                act_layer=act,
                major_radius=cfg.data.major_radius,
                minor_radius=cfg.data.minor_radius,
            )
        elif cfg.model.arch == "constrained_rmcl":
            model = ConstrainedMlpRmclV2(
                in_features=2,
                hidden_features=cfg.model.hidden_features,
                out_features=2,
                n_layers=cfg.model.layers,
                act_layer=act,
                major_radius=cfg.data.major_radius,
                minor_radius=cfg.data.minor_radius,
                n_hyp=cfg.multi_hyp.nsamples,
                beta=cfg.model.beta,
            )
        else:
            raise ValueError(
                "Possible 'arch' values are 'mlp' and 'constrained'."
                f"Got {cfg.model.arch}."
            )
    else:
        if cfg.model.arch == "mlp":
            model = Mlp(
                in_features=1,
                hidden_features=cfg.model.hidden_features,
                out_features=2,
                n_layers=cfg.model.layers,
                act_layer=act,
            )
        elif cfg.model.arch == "constrained":
            model = ConstrainedMlp(
                in_features=1,
                hidden_features=cfg.model.hidden_features,
                out_features=1,
                n_layers=cfg.model.layers,
                act_layer=act,
                radius=cfg.data.radius,
            )
        elif cfg.model.arch == "constrained_rmcl":
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
                f"Got {cfg.model.arch}."
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

    # create MLFlow experiment
    mlf.set_tracking_uri(cfg.run.mlflow_uri)
    mlf.set_experiment(cfg.run.experiment)

    trainer = Trainer(
        model=model,
        optim_cls=optim_cls,
        sched_cls=sched_cls,
        checkpointing_dir=output_dir,
        lr=cfg.train.lr,
        config_train=cfg.train,
        device=device,
        config_data=cfg.data,
    )

    # start new MLFlow run
    with mlf.start_run():
        log_params_from_omegaconf_dict(cfg)

        # TODO: Could add checkpoint loading

        # train
        if cfg.run.train:
            trainer.train(
                epochs=cfg.train.epochs,
                loader=train_loader,
                loss_func=F.mse_loss,
                val_data=all_datasets.validation_set,
            )
            save_and_log_np_artifact(
                output_dir / "train_loss.npy",
                np.array(trainer.loss_list),
            )

        # evaluate
        if cfg.run.test:
            if '3D' in cfg.data.scenario :
                (val_mpjpe, test_mpjpe), (_, test_predictions), _, (_,inout),(_,pdf_list) = trainer.eval_3d(
                eval_sets=(all_datasets.validation_set, all_datasets.test_set),
                metric=calc_mpjpe_3D,
                distribution=distribution,
                major_radius=cfg.data.major_radius,
                minor_radius=cfg.data.minor_radius,
            )
                if cfg.model.arch == "mlp" :
                    std_length_joint = std_length(test_predictions, joint_prediction=True, major_radius=cfg.data.major_radius, minor_radius=cfg.data.minor_radius)
                else : 
                    if "rmcl" in cfg.model.arch:
                        std_length_joint = std_length(test_predictions, joint_prediction=False, mcl_version=True, major_radius=cfg.data.major_radius, minor_radius=cfg.data.minor_radius)
                    else :
                        std_length_joint = std_length(test_predictions, joint_prediction=False, major_radius=cfg.data.major_radius, minor_radius=cfg.data.minor_radius)

                mlf.log_metric(key="std_length_joint", value=std_length_joint)
                mlf.log_metric(key="val_mpjpe", value=val_mpjpe)
                mlf.log_metric(key="test_mpjpe", value=test_mpjpe)

                warning( "std_length_joint : " + str(std_length_joint) + "\n" +
                            "val_mpjpe : " + str(val_mpjpe) + "\n" +
                            "test_mpjpe : " + str(test_mpjpe) + "\n"
                            )

                save_and_log_np_artifact(
                    output_dir / "inout.npy",
                    inout,
                )

                save_and_log_np_artifact(
                    output_dir / "pdf_list.npy",
                    pdf_list,
                )
            else:
                (val_mpjpe, test_mpjpe), (_, test_predictions), _ = trainer.eval(
                    eval_sets=(all_datasets.validation_set, all_datasets.test_set),
                    metric=calc_mpjpe,
                )
                (val_dtc, test_dtc), _, _ = trainer.eval(
                    eval_sets=(all_datasets.validation_set, all_datasets.test_set),
                    metric=distance_to_circle,
                )
                mlf.log_metric(key="val_mpjpe", value=val_mpjpe)
                mlf.log_metric(key="test_mpjpe", value=test_mpjpe)
                mlf.log_metric(key="val_dtc", value=val_dtc)
                mlf.log_metric(key="test_dtc", value=test_dtc)

                warning("val_mpjpe : " + str(val_mpjpe) + "\n" +
                        "test_mpjpe : " + str(test_mpjpe) + "\n" +
                        "val_dtc : " + str(val_dtc) + "\n" +
                        "test_dtc : " + str(test_dtc) + "\n")

                # optionally call function plotting the evaluation results on
                # circle + saving + logging
                plot_predictions(
                    distribution=distribution,
                    X_test=all_datasets.X_test.cpu().numpy(),
                    Y_test=all_datasets.Y_test.cpu().numpy(),
                    predictions_dict={
                        cfg.model.arch: test_predictions.cpu().numpy()
                    },
                    save_path=output_dir / "predictions_plot.png",
                    log_to_mlf=True,
                )

            # save and log predictions
            save_and_log_np_artifact(
                output_dir / "test_predictions.npy",
                test_predictions
            )

            # plot training, save figure and log artifact
            plot_training_curve(
                trainer=trainer,
                save_path=output_dir / "training.png",
                log_to_mlf=True,
            )

    return val_mpjpe

if __name__ == "__main__":
    main()
