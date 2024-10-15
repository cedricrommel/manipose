from typing import Union, List, Dict, Tuple

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from mh_so3_hpe.data import Skeleton
from mh_so3_hpe.metrics import mpjpe_error
from mh_so3_hpe.augmentations import pose_flip
from mh_so3_hpe.architectures import RMCLManifoldMixSTE


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: Union[str, torch.DeviceObjType],
    config: DictConfig,
    skeleton: Skeleton,
    return_hyps: bool = False,
    compute_oracle: bool = True,
) -> Tuple[List, List, Dict]:
    with torch.no_grad():
        model.eval()
        mpjpe_total = 0

        all_target = []
        all_predictions = []

        n = 0
        m_p3d_h36 = 0

        # We only compute oracle metrics for rMCL
        compute_oracle = compute_oracle and isinstance(
            model,
            RMCLManifoldMixSTE
        )
        if compute_oracle:
            oracle_mpjpe_total = 0
            psoracle_mpjpe_total = 0
            all_oracle_preds = []
        with tqdm(loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, (input_2d, target_3d) in enumerate(it, start=1):

                # (B,L,J,C)
                batch_size, L, J, _ = target_3d.shape

                target_3d = target_3d.to(device).float()
                predictions = model(input_2d.to(device).float())
                # For evaluation of RMCL, we compute the expected pose by
                # aggregating the hypothesis
                if isinstance(model, RMCLManifoldMixSTE):
                    hypothesis = model.concat_hyp_and_scores(*predictions)
                    predictions = model.aggregate(*predictions)
                    if compute_oracle:
                        oracle_unagg_mpjpe, oracle_preds = model.aggregate(
                            hypothesis=hypothesis[..., :-1],
                            ground_truth=target_3d,
                            mode="oracle",
                        )
                        oracle_mpjpe = oracle_unagg_mpjpe.sum()
                        oracle_mpjpe /= J
                        psoracle_preds = model.aggregate(
                            hypothesis=hypothesis[..., :-1],
                            scores=hypothesis[..., -1],
                            mode="best_score",
                        )
                        psoracle_mpjpe = mpjpe_error(
                            psoracle_preds,
                            target_3d,
                            mode="sum",
                        )
                        psoracle_mpjpe /= J

                # Test-time augmentation (with flipping)
                if config.train.tta:
                    input_2d_flipped = pose_flip(
                        poses_tuple=(input_2d,),
                        skeleton=skeleton,
                    )[0]
                    predictions_flipped = model(input_2d_flipped.to(device))
                    if isinstance(model, RMCLManifoldMixSTE):
                        # Need to aggregate flipped prediction in rMCL case
                        hypothesis_flipped = model.concat_hyp_and_scores(
                            *predictions_flipped
                        )
                        predictions_flipped = model.aggregate(
                            *predictions_flipped
                        )

                        # Flipped oracle computation and averaging
                        if compute_oracle:
                            hypothesis_flipped[..., :-1] = pose_flip(
                                poses_tuple=(hypothesis_flipped[..., :-1],),
                                skeleton=skeleton,
                            )[0]
                            _, oracle_preds_flipped = model.aggregate(
                                hypothesis=hypothesis_flipped[..., :-1],
                                ground_truth=target_3d,
                                mode="oracle",
                            )
                            oracle_preds = (
                                oracle_preds + oracle_preds_flipped
                            ) / 2

                            oracle_mpjpe = mpjpe_error(
                                oracle_preds,
                                target_3d,
                                mode="sum",
                            )

                            # dividing by J to have same normalization as
                            # non TTA case
                            oracle_mpjpe /= J

                            psoracle_preds_flipped = model.aggregate(
                                hypothesis=hypothesis_flipped[..., :-1],
                                scores=hypothesis_flipped[..., -1],
                                mode="best_score",
                            )

                            psoracle_preds_tta = (
                                psoracle_preds + psoracle_preds_flipped
                            ) / 2

                            psoracle_mpjpe = mpjpe_error(
                                psoracle_preds_tta,
                                target_3d,
                                mode="sum",
                            )

                            # dividing by J to have same normalization as
                            # non TTA case
                            psoracle_mpjpe /= J

                    predictions_flipped = pose_flip(
                        poses_tuple=(predictions_flipped,),
                        skeleton=skeleton,
                    )[0]
                    predictions = (predictions + predictions_flipped) / 2

                n += batch_size

                (
                    renorm_pred_pose,
                    mpjpe_current,
                    mpjpe_p3d_h36,
                ) = evaluation_metrics(
                    pred_pose=predictions,
                    target_3d=target_3d,
                )

                return_hyps = (
                    return_hyps and isinstance(model, RMCLManifoldMixSTE)
                )
                if return_hyps:
                    hypothesis[..., :-1] *= 1000  # scaling only hyps to mm
                    all_predictions.append(hypothesis)
                else:
                    all_predictions.append(renorm_pred_pose)


                mpjpe_total += mpjpe_current.item()
                m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()
                if compute_oracle:
                    oracle_mpjpe_total += oracle_mpjpe
                    psoracle_mpjpe_total += psoracle_mpjpe
                    oracle_preds *= 1000
                    all_oracle_preds.append(oracle_preds)

                all_target.append(target_3d)

                it.set_postfix(
                    ordered_dict={
                        "average_mpjpe": mpjpe_total / batch_no,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            print("Average MPJPE:", mpjpe_total / batch_no)

            performance = m_p3d_h36 / (n * L * J)

            if not compute_oracle:
                return all_predictions, all_target, performance
            else:
                # Oracle metrics were already normalized by J, so only need to
                # divide by n and L. They alse weren't converted to mm.
                oracle_mpjpe_total /= (n * L)
                oracle_mpjpe_total *= 1000
                psoracle_mpjpe_total /= (n * L)
                psoracle_mpjpe_total *= 1000
                return (
                    all_predictions,
                    all_target,
                    performance,
                    oracle_mpjpe_total,
                    psoracle_mpjpe_total,
                    all_oracle_preds,
                )


def evaluation_metrics(
    pred_pose: torch.Tensor,
    target_3d: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    mpjpe_p3d_h36 = mpjpe_error(
        pred_pose,
        target_3d,
        mode="sum",
    )

    mpjpe_current = mpjpe_error(
        pred_pose,
        target_3d,
        mode="average",
    )

    return pred_pose * 1000, mpjpe_current * 1000, mpjpe_p3d_h36 * 1000


def lift_action(
    data_loader,
    model,
    device,
    config,
    skeleton,
    return_hyps,
):
    predictions = evaluate(
        model=model,
        loader=data_loader,
        device=device,
        config=config,
        skeleton=skeleton,
        return_hyps=return_hyps,
    )[0]
    predictions = torch.cat(predictions, dim=0)
    predictions = predictions.detach().cpu().numpy()
    if len(predictions.shape) == 4:
        N, L, J, _ = predictions.shape
        predictions = predictions.reshape(N * L, J, 3) / 1000
    else:
        predictions = np.transpose(predictions, (0, 2, 1, 3, 4))
        N, L, _, J, _ = predictions.shape
        predictions = predictions.reshape(N * L, -1, J, 4)
        predictions[..., :-1] /= 1000

    return predictions
