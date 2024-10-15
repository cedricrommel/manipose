from typing import Union, Optional

import torch
import torch.nn as nn

from omegaconf import DictConfig

from .diff_mlp import DiffMlp
from .conditioners import RawCond
from .utils import compute_noise_scheduling


class LiftingDiffusionModel(nn.Module):
    def __init__(
        self,
        config: DictConfig,
        act: nn.Module,
        device: Union[str, torch.DeviceObjType]
    ):
        super().__init__()
        self.device = device

        config_diff = config.diffusion

        if config_diff.conditioning == "raw":
            self.conditioning = RawCond(
                mix_mode=config_diff.cond_mix_mode,
            )
        else:
            raise ValueError(
                "Invalid value for conditioning param."
            )
        self.conditioning.to(self.device)

        # Whether to predict the whole 3D pose or only the depth
        self.n_output_coords = 2

        if config.model.arch == "mlp":
            self.diffmodel = DiffMlp(
                num_diff_steps=config_diff.num_steps,
                in_features=3,
                hidden_features=config.model.hidden_features,
                out_features=2,
                n_layers=config.model.layers,
                act_layer=act,
            )
        else:
            raise ValueError(
                "Architecture param could not be recognized: "
                f"{config.model.arch}."
            )

        # for evaluation
        self.nsamples = config.multi_hyp.nsamples

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        (
            self.beta,
            self.alpha,
            self.alpha_hat,
            self.sigma,
        ) = compute_noise_scheduling(
            schedule=config_diff.schedule,
            beta_start=config_diff.beta_start,
            beta_end=config_diff.beta_end,
            num_steps=config_diff.num_steps,
        )

        self.alpha_torch = (
            torch.tensor(self.alpha)
            .float()
            .to(self.device)
            .unsqueeze(1)
        )

    def calc_loss_valid(
        self,
        data_1d: torch.Tensor,
        data_2d: torch.Tensor,
        is_train: bool,
    ) -> torch.Tensor:
        loss_sum = 0
        for t in range(0, self.num_steps, 10):  # calculate loss for fixed t
            loss = self.calc_loss(data_1d, data_2d, is_train, set_t=t)
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self,
        data_1d: torch.Tensor,
        data_2d: torch.Tensor,
        is_train: bool,
        set_t: int = -1,
    ) -> torch.Tensor:
        B, _ = data_2d.shape
        if is_train == 1:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        else:
            # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        current_alpha = self.alpha_torch[t].to(self.device)  # (B,1)

        noise = torch.randn_like(data_2d).to(self.device)  # (B,2)
        noisy_data = (current_alpha**0.5) * data_2d + (
            1.0 - current_alpha
        ) ** 0.5 * noise  # (B,2)

        total_input = self.conditioning(
            noisy_data,
            data_1d,
        )

        predicted = self.diffmodel(total_input, t)  # (B,2)
        residual = noise - predicted
        loss = (residual**2).mean()

        return loss

    def forward(
        self,
        data_1d: torch.Tensor,
        n_samples: Optional[int] = None,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        if n_samples is None:
            n_samples = self.nsamples

        B, D = data_1d.shape
        assert D == 1, f"Expected input dimension 1, got {D}."

        if n_steps is None:
            n_steps = self.num_steps

        predicted_samples = torch.zeros(B, n_samples, 2).to(self.device)

        for i in range(n_samples):
            target_2d = torch.randn(
                (B, self.n_output_coords),
                device=data_1d.device,
            )

            for t in range(n_steps - 1, -1, -1):
                diff_input = self.conditioning(
                    target_2d,
                    data_1d,
                )

                # Conditional noise prediction
                predicted_noise = self.diffmodel(
                    diff_input, torch.tensor([t]).to(self.device)
                )  # (B, 2)

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_mean = coeff1 * (target_2d - coeff2 * predicted_noise)

                if t > 0:
                    noise = torch.randn_like(current_mean)
                    target_2d = current_mean + self.sigma[t - 1] * noise

            predicted_samples[:, i] = current_mean.detach()
        return predicted_samples

    def aggregate(
        self,
        hypothesis: torch.Tensor,
        mode: str = "average",
    ) -> torch.Tensor:
        if mode == "average":
            return hypothesis.mean(dim=1)
        else:
            raise ValueError(
                "Implemented aggregation strategies are 'average'."
                f"Got {mode}."
            )
