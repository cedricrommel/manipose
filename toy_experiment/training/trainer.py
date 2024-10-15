from typing import Callable, Tuple, List, Optional, Union
from pathlib import Path
from logging import warning

from tqdm import tqdm
import mlflow as mlf
import numpy as np
from omegaconf import DictConfig

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor

from .averager import AverageMeter
from models import LiftingDiffusionModel, ConstrainedMlpRmcl
from utils.utils import save_state


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        checkpointing_dir: Union[str, Path],
        config_train: DictConfig,
        optim_cls: torch.optim.Optimizer = torch.optim.Adam,
        sched_cls: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        lr: float = 0.001,
        device: Union[torch.device, str] = "cpu",
    ):
        self.model = model

        # store whether model is diffusion or not
        self.diff_enabled = False
        if isinstance(model, LiftingDiffusionModel):
            self.diff_enabled = True

        # store whether MCL or not
        self.mcl_enabled = False
        if isinstance(model, ConstrainedMlpRmcl):
            self.mcl_enabled = True

        self.lr = lr
        self.checkpointing_dir = Path(checkpointing_dir)

        self.device = device
        self.model = self.model.to(device)

        self.optim = optim_cls(model.parameters(), lr=lr)
        if sched_cls is None:
            self.scheduler = None
        else:
            self.scheduler = sched_cls(
                optimizer=self.optim,
                mode="min",
                factor=0.5,
                min_lr=config_train.lr_min,
                patience=config_train.lr_patience,
                threshold=config_train.lr_threshold,
            )

    def reset_metrics(self):
        self.loss_list = list()
        self.val_loss_list = list()
        self.loss_accum = AverageMeter()

    def train(
        self,
        epochs: int,
        loader: DataLoader,
        loss_func: Callable,
        val_data: TensorDataset = None,
        log_in_mlf: bool = True,
    ):
        warning_emitted = False

        if not hasattr(self, "loss_list") or not hasattr(self, "loss_accum"):
            self.reset_metrics()

        best_val_loss = np.inf

        for e in range(1, epochs + 1):
            it = tqdm(loader, desc=f"Epoch {e}")
            self.loss_accum.reset()
            for x, y in it:
                x = x.to(self.device)
                y = y.to(self.device)

                self.optim.zero_grad()

                # compute loss depending on whether we are using diffusion
                if self.diff_enabled:
                    if not warning_emitted:
                        warning(
                            "Argument loss_func ignored for diffusion models. "
                            "Using custom loss."
                        )
                        warning_emitted = True
                    loss = self.model.calc_loss(x, y, is_train=True).mean()
                elif self.mcl_enabled:
                    if not warning_emitted:
                        warning(
                            "Argument loss_func ignored for rMCL models. "
                            "Using custom loss."
                        )
                        warning_emitted = True
                    multi_pred = self.model(x)
                    loss = self.model.wta_with_scoring_l2_loss(multi_pred, y)
                else:
                    pred = self.model(x)
                    loss = loss_func(pred, y)

                loss.backward()
                self.optim.step()
                self.loss_accum.update(
                    val=loss.clone().detach().cpu().numpy().item(),
                    n=loader.batch_size,
                )

            # store and log average training loss across the epoch
            self.loss_list.append(self.loss_accum.avg)
            if log_in_mlf:
                mlf.log_metric(
                    key="loss",
                    value=self.loss_accum.avg,
                    step=e,
                )

            # earlystopping
            if val_data is not None:
                # compute validation loss
                val_loss = self.eval((val_data,), loss_func)[0][0].item()
                self.val_loss_list.append(val_loss)

                # log it
                if log_in_mlf:
                    mlf.log_metric(
                        key="val_loss",
                        value=val_loss,
                        step=e,
                    )

                if val_loss < best_val_loss:
                    # update reference
                    best_val_loss = val_loss

                    # checkpoint model
                    self.save_state(
                        epoch_no=e,
                        log_in_mlf=log_in_mlf,
                        tag="best_val",
                    )

                    # log when validation loss improves
                    if log_in_mlf:
                        mlf.log_metrics(
                            {
                                "best_epoch_loss": e,
                                "best_val_loss": val_loss,
                            },
                            step=e,
                        )

                if self.scheduler is not None:
                    self.scheduler.step(best_val_loss)

        # reload best checkpoint according to validation loss
        if (self.checkpointing_dir / "model_best_val.pth").exists():
            self.model.load_state_dict(
                torch.load(self.checkpointing_dir / "model_best_val.pth")
            )

    def eval(
        self, eval_sets: Tuple[TensorDataset], metric: Callable,
    ) -> Tuple[List[float], List[Tensor]]:
        predictions = list()
        performances = list()
        hypothesis_list = (
            list() if self.diff_enabled or self.mcl_enabled else None
        )
        with torch.no_grad():
            for eval_set in eval_sets:
                assert isinstance(eval_set, TensorDataset)
                X_eval, y_eval = eval_set.tensors
                X_eval = X_eval.to(self.device)
                y_eval = y_eval.to(self.device)

                # compute predictions
                if self.diff_enabled or self.mcl_enabled:
                    hypothesis = self.model(X_eval)
                    hypothesis_list.append(hypothesis)
                    preds = self.model.aggregate(hypothesis)
                else:
                    preds = self.model(X_eval)

                predictions.append(preds)
                try:
                    perf = metric(preds, y_eval)
                except TypeError:
                    perf = metric(preds)

                performances.append(perf)

        return performances, predictions, hypothesis_list

    def save_state(self, epoch_no, log_in_mlf, tag):
        save_state(
            model=self.model,
            optimizer=self.optim,
            scheduler=self.scheduler,
            epoch_no=epoch_no,
            foldername=self.checkpointing_dir,
            log_in_mlf=log_in_mlf,
            tag=tag,
        )
