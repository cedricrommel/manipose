import os
import pickle
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from logging import warning
from pathlib import Path
from typing import Union

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from eval_utils import evaluate, lift_action
from hydra.utils import get_original_cwd
from mup import make_base_shapes, set_base_shapes
from mup.optim import MuAdam
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from mh_so3_hpe.architectures import ManifoldMixSTE, MixSTE, RMCLManifoldMixSTE
from mh_so3_hpe.architectures.utils.mup_utils import mu_init_params
from mh_so3_hpe.augmentations import PoseFlip
from mh_so3_hpe.data import PoseSequenceGenerator, Skeleton
from mh_so3_hpe.data.h36m_lifting import (TEST_SUBJECTS, TRAIN_SUBJECTS,
                                          Human36mDataset)
from mh_so3_hpe.data.utils import create_2d_data, fetch, read_3d_data
from mh_so3_hpe.metrics import (STANDARD_H36M_WEIGHTS, coordwise_error,
                                jointwise_error, jointwise_mse,
                                mean_velocity_error, mse_error, p_mpjpe,
                                sagittal_symmetry, sagittal_symmetry_per_bone,
                                segments_len_err,
                                segments_max_diff_strech_per_bone,
                                segments_max_strech_per_bone,
                                segments_time_consistency,
                                segments_time_consistency_per_bone,
                                smoothness_regularization, weighted_mpjpe_loss,
                                weighted_mse_loss,
                                wta_l2_loss_and_activate_head,
                                wta_with_scoring_loss)
from mh_so3_hpe.utils import (log_metric_to_mlflow, log_metrics_to_mlflow,
                              log_param_to_mlf, log_params_from_omegaconf_dict,
                              seed_worker, set_random_seeds)
from mh_so3_hpe.visualization import (prep_data_for_viz,
                                      prepare_prediction_for_viz,
                                      render_animation)


def save_csv_log(
    output_dir,
    head,
    value,
    is_create=False,
    file_name="test",
    log_in_mlf=False,
):
    if len(value.shape) < 2:
        value = np.expand_dims(value, axis=0)
    df = pd.DataFrame(value)
    file_path = f"{output_dir}/{file_name}.csv"
    if not os.path.exists(file_path) or is_create:
        df.to_csv(file_path, header=head, index=False)
    else:
        with open(file_path, "a") as f:
            df.to_csv(f, header=False, index=False)
    if log_in_mlf:
        mlf.log_artifact(file_path)


def save_state(
    model,
    optimizer,
    scheduler,
    epoch_no,
    foldername,
    log_in_mlf=False,
    tag=None,
):
    if tag is not None:
        tag = f"_{tag}"
    else:
        tag = ""

    params = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch_no,
    }
    torch.save(model.state_dict(), f"{foldername}/model{tag}.pth")
    torch.save(params, f"{foldername}/params{tag}.pth")
    if log_in_mlf:
        mlf.log_artifact(f"{foldername}/model{tag}.pth")
        mlf.log_artifact(f"{foldername}/params{tag}.pth")


def make_loss(config_train, model, skeleton):
    weights = STANDARD_H36M_WEIGHTS if config_train.w_loss else None
    use_vel_loss = config_train.vel_loss > 0
    use_smooth_reg = config_train.smooth_reg > 0
    use_rigid_seg_reg = config_train.rigid_seg_reg > 0
    use_smooth_reg = config_train.smooth_reg > 0
    use_wta_loss = isinstance(model, RMCLManifoldMixSTE)

    if config_train.lat_sym_regularization > 0:
        warning("Lateral symmetry regularization is not implemented yet!")

    # Create dict with loss func components for logging
    loss_terms = {}
    if not use_wta_loss:
        time_axis = 1
        if config_train.sq_loss:
            loss_terms["wloss"] = (
                partial(weighted_mse_loss, weights=weights),
                2
            )
        else:
            loss_terms["wloss"] = (
                partial(weighted_mpjpe_loss, weights=weights),
                2
            )
    else:
        time_axis = 2

        def wloss(hypothesis, y):
            unagg_wta_loss, _ = wta_l2_loss_and_activate_head(
                hypothesis=hypothesis,
                y=y,
                weights=weights,
                squared=config_train.sq_loss,
            )
            return unagg_wta_loss.mean()
        loss_terms["wloss"] = (wloss, 2)

        def score_reg(hypothesis, scores, y):
            _, scoring_loss = wta_with_scoring_loss(
                hypothesis=hypothesis,
                scores=scores,
                y=y,
                beta=config_train.rmcl_score_reg,
                weights=weights,
                squared=config_train.sq_loss,
            )

            return scoring_loss
        loss_terms["score_reg"] = (score_reg, 3)

    if use_vel_loss:
        def vloss(preds, targets):
            return config_train.vel_loss * mean_velocity_error(
                predicted=preds,
                target=targets,
                squared=config_train.sq_loss,
                axis=time_axis,
            )
        loss_terms["vloss"] = (vloss, 2)

    if use_smooth_reg:
        def sreg(preds):
            return config_train.smooth_reg * smoothness_regularization(
                prediction=preds,
                weights=weights,
                axis=time_axis,
            )
        loss_terms["sreg"] = (sreg, 1)
    if use_rigid_seg_reg:
        def rigid_seg_reg(preds):
            return config_train.rigid_seg_reg * segments_time_consistency(
                preds.permute(0, 3, 2, 1),
                skeleton=skeleton,
                mode="sum",
            )
        loss_terms["rigid_seg_reg"] = (rigid_seg_reg, 1)
    return loss_terms


def compute_and_acc_loss(
    prediction,
    y,
    loss_terms,
    avg_loss_terms,
    rmcl=False,
    val=False,
):
    if rmcl:
        prediction, scores = prediction
    loss = torch.zeros(1, device=y.device)
    for loss_term, (term_func, n_inputs) in loss_terms.items():
        if n_inputs == 1:
            loss_term_val = term_func(prediction)
        elif n_inputs == 3 and rmcl:
            loss_term_val = term_func(prediction, scores, y)
        elif n_inputs == 2:
            loss_term_val = term_func(prediction, y)
        else:
            raise ValueError(
                "Unexpected n_inputs in loss term computation"
                f": {n_inputs}."
            )
        loss += loss_term_val

        # Store and accumulate loss terms for logging
        key = f"val_{loss_term}" if val else loss_term
        avg_loss_terms[key] += loss_term_val.item()
    return loss


def train(
    model: nn.Module,
    config: DictConfig,
    device: Union[str, torch.DeviceObjType],
    train_loader: DataLoader,
    valid_loader: DataLoader,
    skeleton: Skeleton,
    foldername: str = "",
    mlflow_on: bool = False,
):
    cfg_train = config.train
    valid_epoch_interval = cfg_train.valid_epoch_interval
    mpjpe_epoch_interval = cfg_train.mpjpe_epoch_interval
    load_state = config.run.checkpoint_params != ""

    if config.model.mup:
        optimizer = MuAdam(
            model.parameters(),
            lr=cfg_train.lr,
            weight_decay=1e-6
        )
    else:
        optimizer = Adam(
            model.parameters(),
            lr=cfg_train.lr,
            weight_decay=1e-6
        )
    if load_state:
        optimizer.load_state_dict(
            torch.load(config.run.checkpoint_params)["optimizer"]
        )

    lr_scheduler_type = cfg_train.lr_scheduler
    if lr_scheduler_type == "cosine":
        T_max = cfg_train.epochs // cfg_train.n_annealing
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=cfg_train.lr_min,
        )
    elif lr_scheduler_type == "plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.5,
            min_lr=cfg_train.lr_min,
            patience=cfg_train.lr_patience,
            threshold=cfg_train.lr_threshold,
        )
    else:
        raise ValueError(
            "Accepted lr_scheduler values are 'cosine' and 'plateau'."
            f"Got {lr_scheduler_type}."
        )

    if load_state:
        lr_scheduler.load_state_dict(
            torch.load(config.run.checkpoint_params)["scheduler"]
        )

    loss_terms = make_loss(cfg_train, model, skeleton=skeleton)

    train_loss = []
    valid_loss = []
    train_loss_epoch = []
    valid_loss_epoch = []

    best_valid_loss = 1e10
    best_mpjpe = 1e10
    best_oracle_mpjpe = 1e10
    best_psoracle_mpjpe = 1e10
    start_epoch = 0
    if load_state:
        start_epoch = torch.load(config.run.checkpoint_params)["epoch"]

    for epoch_no in range(start_epoch, cfg_train.epochs):
        avg_loss = 0
        # Counters for the other training metrics
        avg_loss_terms = defaultdict(int)
        model.train()
        optimizer.zero_grad()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, (X, y) in enumerate(it, start=1):
                X, y = X.to(device), y.to(device)  # B, L, J, 2
                prediction = model(X)

                # Compute each term in the loss and add them
                loss = compute_and_acc_loss(
                    prediction=prediction,
                    y=y,
                    loss_terms=loss_terms,
                    avg_loss_terms=avg_loss_terms,
                    rmcl=isinstance(model, RMCLManifoldMixSTE),
                )

                loss.backward()
                avg_loss += loss.item()

                optimizer.step()
                optimizer.zero_grad()

                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )

        # Average loss terms as well for logging
        epoch_loss_tr = avg_loss / batch_no
        for term, val in avg_loss_terms.items():
            avg_loss_terms[term] = val / batch_no

        train_loss.append(epoch_loss_tr)
        train_loss_epoch.append(epoch_no)

        metrics_to_log = {
            "tr_loss": epoch_loss_tr,
        }
        metrics_to_log.update(avg_loss_terms)

        if (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            avg_loss_terms_valid = defaultdict(int)
            with torch.no_grad():
                with tqdm(
                    valid_loader, mininterval=5.0, maxinterval=50.0
                ) as it:
                    for batch_no, (X_val, y_val) in enumerate(it, start=1):
                        X_val, y_val = X_val.to(device), y_val.to(device)
                        prediction = model(X_val)

                        loss = compute_and_acc_loss(
                            prediction=prediction,
                            y=y_val,
                            loss_terms=loss_terms,
                            avg_loss_terms=avg_loss_terms_valid,
                            rmcl=isinstance(model, RMCLManifoldMixSTE),
                            val=True,
                        )

                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid
                                / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )

            epoch_loss_val = avg_loss_valid / batch_no
            for term, val in avg_loss_terms_valid.items():
                avg_loss_terms_valid[term] = val / batch_no
            valid_loss.append(epoch_loss_val)
            valid_loss_epoch.append(epoch_no)
            metrics_to_log["val_loss"] = epoch_loss_val
            metrics_to_log.update(avg_loss_terms_valid)

            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    epoch_loss_val,
                    "at",
                    epoch_no,
                )
                save_state(
                    model,
                    optimizer,
                    lr_scheduler,
                    epoch_no,
                    foldername,
                    log_in_mlf=mlflow_on,
                    tag="best_val",
                )
                best_model_state = deepcopy(model.state_dict())

                # Log when validation loss improves
                metrics_to_log.update(
                    {
                        "best_epoch_loss": epoch_no,
                        "best_val_loss": epoch_loss_val,
                    }
                )

            if lr_scheduler_type == "plateau":
                lr_scheduler.step(best_valid_loss)
            else:
                lr_scheduler.step()

            if (epoch_no + 1) == cfg_train.epochs:
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.plot(train_loss_epoch, train_loss)
                ax.plot(valid_loss_epoch, valid_loss)
                ax.grid(True)
                plt.show()
                fig.savefig(f"{foldername}/loss.png")

        # Compute MPJPE every desired nb of epochs
        if (epoch_no + 1) % mpjpe_epoch_interval == 0:
            metrics = evaluate(
                model=model,
                loader=valid_loader,
                device=device,
                config=config,
                skeleton=skeleton,
                compute_oracle=True,
            )[2:]
            if isinstance(model, RMCLManifoldMixSTE):
                mpjpe_val, oracle_mpjpe_val, psoracle_mpjpe_val, _ = metrics
                oracle_mpjpe_val = float(oracle_mpjpe_val)
                psoracle_mpjpe_val = float(psoracle_mpjpe_val)
                metrics_to_log["val_oracle_mpjpe"] = oracle_mpjpe_val
                metrics_to_log["val_ps_oracle_mpjpe"] = psoracle_mpjpe_val

                # Log to MLFlow when there is an improvement in oracle and
                # pseudo-oracle
                if best_oracle_mpjpe > oracle_mpjpe_val:
                    best_oracle_mpjpe = oracle_mpjpe_val
                    metrics_to_log.update(
                        {
                            "best_epoch_oracle_mpjpe": epoch_no,
                            "best_val_oracle_mpjpe": oracle_mpjpe_val,
                        }
                    )
                    save_state(
                        model,
                        optimizer,
                        lr_scheduler,
                        epoch_no,
                        foldername,
                        log_in_mlf=mlflow_on,
                        tag="best_oracle_mpjpe",
                    )

                if best_psoracle_mpjpe > psoracle_mpjpe_val:
                    best_psoracle_mpjpe = psoracle_mpjpe_val
                    metrics_to_log.update(
                        {
                            "best_epoch_ps_oracle_mpjpe": epoch_no,
                            "best_val_ps_oracle_mpjpe": psoracle_mpjpe_val,
                        }
                    )
                    save_state(
                        model,
                        optimizer,
                        lr_scheduler,
                        epoch_no,
                        foldername,
                        log_in_mlf=mlflow_on,
                        tag="best_ps_oracle_mpjpe",
                    )

            else:
                mpjpe_val = metrics[0]
            metrics_to_log["val_mpjpe"] = mpjpe_val

            # Log to MLFlow when there is an improvement in average prediction
            if best_mpjpe > mpjpe_val:
                best_mpjpe = mpjpe_val
                metrics_to_log.update(
                    {
                        "best_epoch_mpjpe": epoch_no,
                        "best_val_mpjpe": best_mpjpe,
                    }
                )
                save_state(
                    model,
                    optimizer,
                    lr_scheduler,
                    epoch_no,
                    foldername,
                    log_in_mlf=mlflow_on,
                    tag="best_mpjpe",
                )
                # And save state dict
                best_model_state = deepcopy(model.state_dict())

        # Write all logs into MLflow at once
        log_metrics_to_mlflow(
            metrics_to_log,
            step=epoch_no,
            mlflow_on=mlflow_on,
        )

    save_state(
        model, optimizer, lr_scheduler, cfg_train.epochs, foldername, tag="end"
    )
    np.save(f"{foldername}/train_loss.npy", np.array(train_loss))
    np.save(f"{foldername}/valid_loss.npy", np.array(valid_loss))

    # load best weights
    model.load_state_dict(best_model_state)
    return best_mpjpe, model


def fetch_and_prepare_data(cfg, proj_name):
    data_dir = Path(cfg.data.data_dir)
    preproc_dataset_path = data_dir / (
        f"preproc_data_3d_{cfg.data.dataset}_{cfg.data.joints}_"
        f"{proj_name}.pkl"
    )

    if preproc_dataset_path.exists():
        print("==> Loading preprocessed dataset...")
        with open(preproc_dataset_path, "rb") as f:
            dataset = pickle.load(f)
    else:
        print("==> Loading raw dataset...")
        dataset_path = data_dir / f"data_3d_{cfg.data.dataset}.npz"

        dataset = Human36mDataset(dataset_path, n_joints=cfg.data.joints)

        print("==> Preparing data...")
        dataset = read_3d_data(dataset)

        print("==> Caching data...")
        with open(preproc_dataset_path, "wb") as f:
            pickle.dump(dataset, f)

    print("==> Loading 2D detections...")
    inputs_path = data_dir / (
        f"data_2d_{cfg.data.dataset}_{cfg.data.keypoints}.npz"
    )
    keypoints = create_2d_data(inputs_path, dataset)
    return keypoints, dataset


def get_subjects_and_actions(dataset, cfg):
    if cfg.data.use_valid:
        subjects_train = TRAIN_SUBJECTS[:-1]
        subjects_val = TRAIN_SUBJECTS[-1:]
    else:
        subjects_train = TRAIN_SUBJECTS
        subjects_val = []
    subjects_test = TEST_SUBJECTS

    if cfg.data.data == "one":
        subjects_train = [subjects_train[0]]

    action_filter = (
        None if cfg.data.actions == "*" else cfg.data.actions.split(",")
    )
    if action_filter is not None:
        action_filter = list(
            map(
                lambda x: dataset.define_actions(x)[0], action_filter
            )
        )
        print("==> Selected actions: {}".format(action_filter))

    return [subjects_train, subjects_val, subjects_test], action_filter


def create_dataloader(
    keypoints,
    dataset,
    action_filter,
    subjects,
    cfg,
    train=True,
):
    poses, poses_2d, _, cameras = fetch(
        subjects,
        dataset,
        keypoints,
        action_filter,
    )
    transform = None
    if cfg.train.flip_aug:
        transform = PoseFlip(skeleton=dataset.skeleton, probability=0.5)

    generator = PoseSequenceGenerator(
        poses,
        poses_2d,
        cameras,
        seq_len=cfg.data.seq_len,
        random_start=train,
        miss_type=cfg.data.miss_type,
        miss_rate=cfg.data.miss_rate,
        transform=transform,
        # drop_last=False,
    )

    data_loader = DataLoader(
        generator,
        batch_size=(
            cfg.train.batch_size if train else cfg.train.batch_size_test
        ),
        shuffle=True if train else False,
        num_workers=cfg.train.workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
    )
    return data_loader


def _instantiate_model(
    cfg: DictConfig,
    skeleton: Skeleton,
) -> nn.Module:
    if cfg.model.arch == "mixste":
        model = MixSTE(
            num_frame=cfg.data.seq_len,
            num_joints=skeleton.num_joints,
            in_chans=2,
            out_dim=3,
            num_heads=cfg.model.nheads,
            depth=cfg.model.layers,
            embed_dim=cfg.model.channels,
            drop_path_rate=cfg.model.drop_path_rate,
            mup=cfg.model.mup,
        )
    elif cfg.model.arch == "manifold":
        model = ManifoldMixSTE(
            skeleton=skeleton,
            num_frame=cfg.data.seq_len,
            num_joints=skeleton.num_joints,
            num_bones=skeleton.num_bones,
            in_chans=2,
            rot_rep_dim=cfg.model.rot_dim,
            num_heads_rot=cfg.model.nheads,
            depth_rot=cfg.model.layers,
            embed_dim_rot=cfg.model.channels,
            num_heads_seg=cfg.model.nheads_seg,
            depth_seg=cfg.model.layers_seg,
            embed_dim_seg=cfg.model.channels_seg,
            drop_path_rate=cfg.model.drop_path_rate,
            mup=cfg.model.mup,
        )
    elif cfg.model.arch == "rmcl_manifold":
        model = RMCLManifoldMixSTE(
            skeleton=skeleton,
            num_frame=cfg.data.seq_len,
            num_joints=skeleton.num_joints,
            num_bones=skeleton.num_bones,
            in_chans=2,
            rot_rep_dim=cfg.model.rot_dim,
            num_heads_rot=cfg.model.nheads,
            depth_rot=cfg.model.layers,
            embed_dim_rot=cfg.model.channels,
            num_heads_seg=cfg.model.nheads_seg,
            depth_seg=cfg.model.layers_seg,
            embed_dim_seg=cfg.model.channels_seg,
            drop_path_rate=cfg.model.drop_path_rate,
            n_hyp=cfg.multi_hyp.n_hyp,
            mup=cfg.model.mup,
        )
    else:
        raise ValueError(
            "Only MixSTE, Manifold-MixSTE and RMCL-Manifold-MixSTE implemented"
            f" for now. Got option {cfg.model.arch}."
        )

    return model


def set_mup_base_shapes(model, cfg, skeleton):
    base_shapes_dir = Path(cfg.run.base_shape_dir)
    base_shape_path = base_shapes_dir / (
        f"{cfg.model.arch}_width-seq_scaling_{cfg.multi_hyp.n_hyp}.bsh"
    )
    # If base shape file does not exist, create it
    if not base_shape_path.exists():
        print(f"Creating new base shape at {base_shape_path}")
        base_shapes_dir.mkdir(exist_ok=True)
        copied_cfg = deepcopy(cfg)
        copied_cfg.model.channels = 64
        copied_cfg.model.channels_seg = 64
        copied_cfg.data.seq_len = 27
        base_model = _instantiate_model(cfg=copied_cfg, skeleton=skeleton)
        copied_cfg.model.channels = 128
        copied_cfg.model.channels_seg = 128
        copied_cfg.data.seq_len = 81
        delta_model = _instantiate_model(cfg=copied_cfg, skeleton=skeleton)
        make_base_shapes(base_model, delta_model, base_shape_path)
        del base_model
        del delta_model

    return set_base_shapes(model, str(base_shape_path))


def create_model(
    cfg: DictConfig,
    skeleton: Skeleton,
) -> nn.Module:
    model = _instantiate_model(cfg=cfg, skeleton=skeleton)

    # Set base shapes for muP scaling
    if cfg.model.mup:
        model = set_mup_base_shapes(model, cfg, skeleton)

    return model


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("==> Using settings:")
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: %s" % device)

    cwd = Path(os.getcwd())
    output_dir = cwd / cfg.run.experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load preprocessed or raw dataset
    original_cwd = Path(get_original_cwd())
    proj_name = original_cwd.parents[0].name
    keypoints, dataset = fetch_and_prepare_data(cfg, proj_name=proj_name)

    # Split subjects and prepare actions
    subjects_split, actions = get_subjects_and_actions(dataset, cfg)
    subjects_train, subjects_val, subjects_test = subjects_split

    # Set seeds for init reproducibility
    print(f"==> Setting seeds to {cfg.run.seed} for init")
    set_random_seeds(
        seed=cfg.run.seed,
        cuda=True,
        cudnn_benchmark=cfg.run.cudnn_benchmark,
        set_deterministic=cfg.run.set_deterministic,
    )

    # Creating model
    model = create_model(
        cfg,
        skeleton=dataset.skeleton,
    )

    # DataParallel is incompatibel with muP (could be replaced by
    # DistributedDataParallel if necessary)
    if torch.cuda.device_count() > 1 and not cfg.model.mup:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    if cfg.run.checkpoint_model != "":
        model_path = cwd / cfg.run.checkpoint_model
        checkpoint = torch.load(model_path)
        if "model_pos" in checkpoint.keys():
            checkpoint = checkpoint["model_pos"]
        model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {model_path}!")
    elif cfg.model.mup:
        # Re-initialize model according to muP
        mu_init_params(model)

    mlflow_on = cfg.run.mlflow_on
    if mlflow_on:
        # Lazy import of MLFlow if requested
        import mlflow as mlf
        mlf.set_tracking_uri(cfg.run.mlflow_uri)
        mlf.set_experiment(cfg.run.experiment)

    # Used to log to MLFlow or not depending on config
    context = mlf.start_run if mlflow_on else nullcontext

    with context():
        log_params_from_omegaconf_dict(cfg, mlflow_on=mlflow_on)
        # to facilitate retrival of exp data
        log_param_to_mlf("run.experiment_dir", output_dir, mlflow_on=mlflow_on)
        if cfg.run.train:
            train_loader = create_dataloader(
                keypoints=keypoints,
                dataset=dataset,
                action_filter=actions,
                subjects=subjects_train,
                cfg=cfg,
                train=True,
            )
            print(
                ">>> Training dataset length: {:d}".format(
                    len(train_loader) * cfg.train.batch_size
                )
            )

            valid_loader = create_dataloader(
                keypoints=keypoints,
                dataset=dataset,
                action_filter=actions,
                subjects=subjects_val if cfg.data.use_valid else subjects_test,
                cfg=cfg,
                train=False,
            )
            print(
                ">>> Validation dataset length: {:d}".format(
                    len(valid_loader) * cfg.train.batch_size_test
                )
            )

            best_valid_mpjpe, model = train(
                model=model,
                config=cfg,
                device=device,
                train_loader=train_loader,
                valid_loader=valid_loader,
                skeleton=dataset.skeleton,
                foldername=output_dir,
                mlflow_on=mlflow_on,
            )

        if cfg.run.test:
            if actions is None:
                actions = [
                    "walking",
                    "eating",
                    "smoking",
                    "discussion",
                    "directions",
                    "greeting",
                    "phoning",
                    "posing",
                    "purchases",
                    "sitting",
                    "sittingdown",
                    "photo",
                    "waiting",
                    "walkdog",
                    "walktogether",
                ]

            head = [
                "act",
                "mpjpe",
                "sag sym",
                "seg std",
                "p-mpjpe",
                "mvjpe",
                "mse",
                "err var",
                "seg err",
            ]
            if isinstance(model, RMCLManifoldMixSTE):
                head += ["oracle mpjpe", "pseudo oracle mpjpe"]
                errs = np.zeros([len(actions) + 1, 10])
            else:
                errs = np.zeros([len(actions) + 1, 8])

            analytics = {
                k: (
                    np.zeros(
                        [len(actions) + 1, dataset.skeleton.num_bones]
                    ),
                    ["act", *dataset.skeleton.bones_names],
                )
                for k in [
                    "seg_symmetry",
                    "seg_consistency",
                    "seg_max_strech",
                    "seg_max_delta_strech",
                ]
            }
            analytics["cw_err"] = (
                np.zeros([len(actions) + 1, 3]),
                ["act", "x", "y", "z"],
            )
            analytics["jw_err"] = (
                np.zeros([len(actions) + 1, dataset.skeleton.num_joints]),
                ["act", *dataset.skeleton.joints_names],
            )
            all_seg_errs = []
            all_jw_err_var = []
            # create hyps store
            all_pred_hyps = []

            for i, action in enumerate(actions):
                print(f"Assessing action: {action} - [{i + 1}/{len(actions)}]")
                test_loader = create_dataloader(
                    keypoints=keypoints,
                    dataset=dataset,
                    action_filter=[action],
                    subjects=["S11"],
                    # subjects=subjects_test,
                    cfg=cfg,
                    train=False,
                )
                print(
                    ">>> Test dataset length: {:d}".format(
                        test_loader.__len__() * cfg.train.batch_size_test
                    )
                )

                if isinstance(model, RMCLManifoldMixSTE):
                    (
                        hypotheses,
                        target_poses,
                        mpjpe,
                        o_mpjpe,
                        pso_mpjpe,
                        generated_poses,  # <-- oracle preds
                    ) = evaluate(
                        model=model,
                        loader=test_loader,
                        device=device,
                        config=cfg,
                        skeleton=dataset.skeleton,
                        return_hyps=cfg.viz.hypothesis,
                        compute_oracle=True,
                    )
                    errs[i, 8] = o_mpjpe
                    errs[i, 9] = pso_mpjpe
                else:
                    generated_poses, target_poses, mpjpe = evaluate(
                        model=model,
                        loader=test_loader,
                        device=device,
                        config=cfg,
                        skeleton=dataset.skeleton,
                    )

                # add hyps to store
                all_pred_hyps.append((hypotheses, target_poses))

                errs[i, 0] = mpjpe

                with torch.no_grad():
                    generated_poses = torch.cat(
                        generated_poses, dim=0
                    ).permute(0, 3, 2, 1)
                    errs[i, 1] = (
                        sagittal_symmetry(
                            joints_coords=generated_poses,
                            skeleton=dataset.skeleton,
                            mode="average",
                            squared=False,
                        )
                        .cpu()
                        .numpy()
                    )
                    _, _, J, _ = generated_poses.shape  # B, 3, J, L
                    errs[i, 2] = (
                        segments_time_consistency(
                            joints_coords=(
                                generated_poses.permute(1, 2, 0, 3)  # 3, J, B, L
                                               .reshape(1, 3, J, -1)  # 1, 3, J, B*L
                            ),
                            skeleton=dataset.skeleton,
                            mode="std",
                        )
                        .cpu()
                        .numpy()
                    )

                    target_poses = torch.cat(target_poses, dim=0) * 1000

                    errs[i, 3] = p_mpjpe(
                        predicted=generated_poses.permute(0, 3, 2, 1),
                        target=target_poses,
                    )
                    errs[i, 4] = mean_velocity_error(
                        predicted=generated_poses.permute(0, 3, 2, 1),
                        target=target_poses,
                        squared=False,
                        axis=1,
                    )

                    mse = mse_error(
                        batch_gt=target_poses,
                        batch_imp=generated_poses.permute(0, 3, 2, 1),
                        mode="average"
                    )
                    err_var = mse - mpjpe**2
                    errs[i, 5] = mse
                    errs[i, 6] = err_var

                    errs[i, 7] = segments_len_err(
                        batch_gt=target_poses.permute(0, 3, 2, 1),
                        batch_imp=generated_poses,
                        skeleton=dataset.skeleton,
                        mode="average",
                        signed=False,
                    )

                    seg_errs = segments_len_err(
                        batch_gt=target_poses.permute(0, 3, 2, 1),
                        batch_imp=generated_poses,
                        skeleton=dataset.skeleton,
                        mode="no_agg",
                    )
                    # randomly picking 1000 samples to viz the distribution
                    rand_idx = np.random.randint(
                        low=0,
                        high=seg_errs.shape[0] - 1,
                        size=1000,
                    )
                    all_seg_errs.append(seg_errs[rand_idx, :].cpu().numpy())

                    bw_seg_sym = (
                        sagittal_symmetry_per_bone(
                            joints_coords=generated_poses,
                            skeleton=dataset.skeleton,
                            mode="average",
                            squared=False,
                        )
                        .cpu()
                        .numpy()
                    )

                    analytics["seg_symmetry"][0][
                        i, dataset.skeleton.bones_left
                    ] = bw_seg_sym
                    analytics["seg_symmetry"][0][
                        i, dataset.skeleton.bones_right
                    ] = bw_seg_sym

                    analytics["seg_consistency"][0][i] = (
                        segments_time_consistency_per_bone(
                            joints_coords=generated_poses.permute(1, 2, 0, 3)  # 3, J, B, L
                                               .reshape(1, 3, J, -1),  # 1, 3, J, B*L
                            skeleton=dataset.skeleton,
                            mode="std",
                        )
                        .cpu()
                        .numpy()
                    )

                    analytics["jw_err"][0][i] = (
                        jointwise_error(
                            generated_poses.permute(0, 3, 2, 1),
                            target_poses,
                            "average",
                        )
                        .cpu()
                        .numpy()
                    )
                    analytics["cw_err"][0][i] = (
                        coordwise_error(
                            generated_poses.permute(0, 3, 2, 1),
                            target_poses,
                            "average",
                        )
                        .cpu()
                        .numpy()
                    )
                    jw_mse = jointwise_mse(
                        generated_poses.permute(0, 3, 2, 1),
                        target_poses,
                        "average",
                    ).cpu().numpy()
                    jw_err_var = jw_mse - (analytics["jw_err"][0][i])**2
                    all_jw_err_var.append(jw_err_var)

                    min_seg_len, max_seg_len = segments_max_strech_per_bone(
                        joints_coords=generated_poses.permute(1, 2, 0, 3)  # 3, J, B, L
                                               .reshape(1, 3, J, -1),  # 1, 3, J, B*L
                        skeleton=dataset.skeleton,
                    )
                    analytics["seg_max_strech"][0][i] = (
                        max_seg_len - min_seg_len
                    ).cpu().numpy()

                    # seg_len_diff = torch.abs(
                    #     torch.diff(
                    #         generated_poses.permute(1, 2, 0, 3)  # 3, J, B, L
                    #                         .reshape(1, 3, J, -1),  # 1, 3, J, B*L
                    #         dim=3,
                    #     )
                    # )
                    # _, max_seg_len_delta = segments_max_strech_per_bone(
                    #     joints_coords=seg_len_diff,
                    #     skeleton=dataset.skeleton,
                    # )
                    max_delta_stretch, idx_max_delta_stretch = segments_max_diff_strech_per_bone(
                        joints_coords=generated_poses.permute(1, 2, 0, 3)  # 3, J, B, L
                                            .reshape(1, 3, J, -1),  # 1, 3, J, B*L
                        skeleton=dataset.skeleton
                    )

                    analytics["seg_max_delta_strech"][0][i] = (
                        max_delta_stretch
                    ).cpu().numpy()

                    worst_idx_left_arm = idx_max_delta_stretch[15]
                    worst_idx_right_arm = idx_max_delta_stretch[13]
                    worst_cam_left_arm = worst_idx_left_arm % 4
                    worst_frame_left_arm = worst_idx_left_arm // 4
                    worst_cam_right_arm = worst_idx_right_arm % 4
                    worst_frame_right_arm = worst_idx_right_arm // 4

                    print(
                        f"For actions {action}, "
                        f"worst frames are {worst_frame_left_arm} "
                        f"at cam {worst_cam_left_arm}"
                        f" and frame {worst_frame_right_arm}"
                        f" at cam {worst_cam_right_arm}."
                    )

            errs[-1] = np.mean(errs[:-1], axis=0)
            log_metric_to_mlflow(
                "best_val_mpjpe",
                errs[-1, 0],
                mlflow_on=mlflow_on,
            )
            log_metric_to_mlflow(
                "sag_sym",
                errs[-1, 1],
                mlflow_on=mlflow_on,
            )
            log_metric_to_mlflow(
                "seg_std",
                errs[-1, 2],
                mlflow_on=mlflow_on,
            )
            log_metric_to_mlflow(
                "val_pmpjpe",
                errs[-1, 3],
                mlflow_on=mlflow_on,
            )
            log_metric_to_mlflow(
                "val_mvjpe",
                errs[-1, 4],
                mlflow_on=mlflow_on,
            )
            log_metric_to_mlflow(
                "val_mse",
                errs[-1, 5],
                mlflow_on=mlflow_on,
            )
            log_metric_to_mlflow(
                "val_err_var",
                errs[-1, 6],
                mlflow_on=mlflow_on,
            )
            log_metric_to_mlflow(
                "val_mean_seg_err",
                errs[-1, 7],
            )
            if isinstance(model, RMCLManifoldMixSTE):
                log_metric_to_mlflow(
                    "best_val_oracle_mpjpe",
                    errs[-1, 8],
                    mlflow_on=mlflow_on,
                )
                log_metric_to_mlflow(
                    "best_val_ps_oracle_mpjpe",
                    errs[-1, 9],
                    mlflow_on=mlflow_on,
                )

            actions = np.array(actions + ["average"])[:, None]
            value = np.hstack([actions, errs.astype(np.str)])
            save_csv_log(
                output_dir=output_dir,
                head=head,
                value=value,
                is_create=True,
                file_name="protocol_1_err",
                log_in_mlf=mlflow_on,
            )

            for metric_name, (values, a_head) in analytics.items():
                values[-1] = np.mean(values[:-1], axis=0)
                values = np.hstack([actions, values.astype(np.str)])
                save_csv_log(
                    output_dir=output_dir,
                    head=a_head,
                    value=values,
                    is_create=True,
                    file_name=metric_name,
                    log_in_mlf=mlflow_on,
                )

            all_seg_errs = np.concatenate(all_seg_errs, axis=0)
            seg_errs_path = str(output_dir / "all_seg_errs.npy")
            np.save(seg_errs_path, all_seg_errs)
            all_jw_err_var = np.stack(all_jw_err_var, axis=0)
            jw_err_var_path = str(output_dir / "all_jw_err_var.npy")
            np.save(jw_err_var_path, all_jw_err_var)
            # save predicted hypotheses
            hyps_preds_path = str(output_dir / "all_pred_hyps.pkl")
            with open(hyps_preds_path, "wb+") as f:
                pickle.dump(all_pred_hyps, f)

            if mlflow_on:
                mlf.log_artifact(seg_errs_path)
                mlf.log_artifact(jw_err_var_path)
                mlf.log_artifact(hyps_preds_path)

        if cfg.run.viz:
            figures_dir = cwd / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)

            (
                render_loader,
                input_keypoints,
                ground_truth,
                cam,
            ) = prep_data_for_viz(cfg, dataset=dataset, keypoints=keypoints)

            prediction = lift_action(
                data_loader=render_loader,
                model=model,
                device=device,
                config=cfg,
                skeleton=dataset.skeleton,
                return_hyps=cfg.viz.hypothesis,
            )

            multihyp = (
                cfg.viz.hypothesis and isinstance(model, RMCLManifoldMixSTE)
            )
            prediction = prepare_prediction_for_viz(
                prediction=prediction,
                cam=cam,
                multihyp=multihyp
            )

            ground_truth = prepare_prediction_for_viz(
                prediction=ground_truth,
                cam=cam,
            )
            anim_output = {
                type(model).__name__: prediction,
                "Ground truth": ground_truth,
            }

            if cfg.viz.viz_output != "":
                output_name = cfg.viz.viz_output
            else:
                hyps_tag = "_hyps" if cfg.viz.hypothesis else ""
                output_name = (
                    f"{cfg.model.arch}{hyps_tag}_"
                    f"{cfg.viz.viz_subject}_{cfg.viz.viz_action}_"
                    f"{cfg.viz.viz_camera}.{cfg.viz.extension}"
                )
            output_name = figures_dir / output_name

            print("==> Rendering...")

            render_animation(
                keypoints=input_keypoints,
                poses=anim_output,
                skeleton=dataset.skeleton,
                fps=dataset.fps,
                bitrate=cfg.viz.viz_bitrate,
                azim=cam["azimuth"],
                output=str(output_name),
                limit=cfg.viz.viz_limit,
                downsample=cfg.viz.viz_downsample,
                size=cfg.viz.viz_size,
                input_video_path=cfg.viz.viz_video,
                viewport=(cam["res_w"], cam["res_h"]),
                input_video_skip=cfg.viz.viz_skip,
            )

    # We need to return the validation metric for HP search
    if cfg.run.train:
        return best_valid_mpjpe


if __name__ == "__main__":
    main()
