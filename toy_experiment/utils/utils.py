import os
import numbers
import random
from warnings import warn
from pathlib import Path

import mlflow as mlf
from omegaconf import DictConfig, ListConfig
import numpy as np

import torch


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
        The random state object based on `seed` parameter.
    """
    if seed is None or isinstance(seed, numbers.Integral):
        return np.random.default_rng(seed=seed)
    if isinstance(seed, np.random.Generator):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.Generator instance" % seed
    )


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


def polar2cartesian(r, theta):

    """
    polar22cartesian(r,theta)
    =========================

    This function converts polar coordinates to cartesian coordinates. Note
    that we allow r to be scalar, it will broadcast to the correct size if
    theta is an array.

    INPUT:

        * r - radial coordinate - float scalar or array the same size as theta.
        * theta - angular coordinate - float scalar or array.

    OUTPUT:

        * x, y - Cartesian coordinates - these are arrays of floats the same
        size as theta.

    """

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return x, y


# Source: https://medium.com/optuna/easy-hyperparameter-management-with-hydra-
# mlflow-and-optuna-783730700e7d


def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)


def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f"{parent_name}.{k}", v)
            else:
                mlf.log_param(f"{parent_name}.{k}", v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlf.log_param(f"{parent_name}.{i}", v)


def save_and_log_np_artifact(save_path, data_arr):
    np.save(save_path, data_arr)
    mlf.log_artifact(save_path)


def save_state(
    model, optimizer, scheduler, epoch_no, foldername, log_in_mlf=False,
    tag=None,
):
    foldername = Path(foldername)
    if tag is not None:
        tag = f"_{tag}"
    else:
        tag = ""

    params = {
        "optimizer": optimizer.state_dict(),
        "epoch": epoch_no,
    }
    if scheduler is not None:
        params["scheduler"] = scheduler.state_dict()

    torch.save(model.state_dict(), foldername / f"model{tag}.pth")
    torch.save(params, foldername / f"params{tag}.pth")
    if log_in_mlf:
        mlf.log_artifact(foldername / f"model{tag}.pth")
        mlf.log_artifact(foldername / f"params{tag}.pth")
