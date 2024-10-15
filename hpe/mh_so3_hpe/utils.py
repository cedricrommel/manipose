from __future__ import absolute_import, division

import os
import random
from warnings import warn

import mlflow as mlf
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def lr_decay(optimizer, step, lr, decay_step, gamma):
    lr = lr * gamma ** (step / decay_step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def save_ckpt(state, ckpt_path, suffix=None, log_in_mlf=False):
    if suffix is None:
        suffix = "epoch_{:04d}".format(state["epoch"])

    file_path = os.path.join(ckpt_path, "ckpt_{}.pth.tar".format(suffix))
    torch.save(state, file_path)
    if log_in_mlf:
        import mlflow as mlf
        mlf.log_artifact(file_path)


def set_random_seeds(seed, cuda, cudnn_benchmark=None, set_deterministic=None):
    """Set seeds for python random module numpy.random and torch.
    For more details about reproducibility in pytorch see
    https://pytorch.org/docs/stable/notes/randomness.html

    Parameters
    ----------
    seed: int
        Random seed.
    cuda: bool
        Whether to set cuda seed with torch.
    cudnn_benchmark: bool (default=None)
        Whether pytorch will use cudnn benchmark. When set to `None` it will
        not modify torch.backends.cudnn.benchmark (displays warning in the
        case of possible lack of reproducibility). When set to True, results
        may not be reproducible (no warning displayed). When set to False it
        may slow down computations.
    set_deterministic: bool (default=None)
        Whether to refrain from using non-deterministic torch algorithms.
        If you are using CUDA tensors, and your CUDA version is 10.2 or
        greater, you should set the environment variable
        CUBLAS_WORKSPACE_CONFIG according to CUDA documentation:
        https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    Notes
    -----
    In some cases setting environment variable `PYTHONHASHSEED` may be needed
    before running a script to ensure full reproducibility. See
    https://forums.fast.ai/t/solved-reproducibility-where-is-the-randomness-coming-in/31628/14
    Using this function may not ensure full reproducibility of the results as
    we do not set `torch.use_deterministic_algorithms(True)`.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        if isinstance(cudnn_benchmark, bool):
            torch.backends.cudnn.benchmark = cudnn_benchmark
        elif cudnn_benchmark is None:
            if torch.backends.cudnn.benchmark:
                warn(
                    "torch.backends.cudnn.benchmark was set to True which may"
                    " results in lack of reproducibility. In some cases to "
                    "ensure reproducibility you may need to set "
                    "torch.backends.cudnn.benchmark to False.",
                    UserWarning,
                )
        else:
            raise ValueError(
                "cudnn_benchmark expected to be bool or None, "
                f"got '{cudnn_benchmark}'"
            )
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if set_deterministic is None:
        if torch.backends.cudnn.deterministic is False:
            warn(
                "torch.backends.cudnn.deterministic was set to False which may"
                " results in lack of reproducibility. In some cases to "
                "ensure reproducibility you may need to set "
                "torch.backends.cudnn.deterministic to True.",
                UserWarning,
            )
        else:
            torch.use_deterministic_algorithms(set_deterministic)
            if set_deterministic:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Source: https://medium.com/optuna/easy-hyperparameter-management-with-hydra-
# mlflow-and-optuna-783730700e7d


def log_params_from_omegaconf_dict(params, mlflow_on):
    if mlflow_on:
        for param_name, element in params.items():
            _explore_recursive(param_name, element)


def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f"{parent_name}.{k}", v)
            else:
                log_param_to_mlf(f"{parent_name}.{k}", v, mlflow_on=True)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            log_param_to_mlf(f"{parent_name}.{i}", v, mlflow_on=True)


def log_param_to_mlf(key, value, mlflow_on):
    if mlflow_on:
        import mlflow as mlf
        try:
            mlf.log_param(key=key, value=value)
        except mlf.exceptions.RestException as e:
            print(e)
            pass


def log_metric_to_mlflow(key, value, mlflow_on, step=None):
    if mlflow_on:
        import mlflow as mlf
        try:
            mlf.log_metric(
                key=key,
                value=value,
                step=step,
            )
        except mlf.exceptions.RestException as e:
            print(e)
            pass


def log_metrics_to_mlflow(metrics, mlflow_on, step=None):
    if mlflow_on:
        import mlflow as mlf
        try:
            mlf.log_metrics(
                metrics=metrics,
                step=step,
            )
        except mlf.exceptions.RestException as e:
            print(e)
            pass
