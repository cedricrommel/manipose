import torch
from torch import nn


def mix_data_with_condition(
    noisy_data,
    data_2d,
    mix_mode="concat",
):
    if mix_mode == "concat" or mix_mode == "z_only":
        # Concatenate over channel dim
        return torch.cat([noisy_data, data_2d], dim=1)
    elif mix_mode == "sum":
        assert torch.is_same_size(noisy_data, data_2d), (
            "noisy_data and data_2d need to have the same size to be summed."
            f"Got {noisy_data.shape} and {data_2d.shape}."
        )
        # Sum
        return noisy_data + data_2d
    else:
        raise ValueError(
            "Accepted mix_mode values are 'sum', 'concat' and 'z_only'."
            f"Got {mix_mode}."
        )


class RawCond(nn.Module):
    def __init__(self, mix_mode="concat") -> None:
        super().__init__()
        self.mix_mode = mix_mode
        self.out_dim = 3 if mix_mode == "concat" else 2

    def forward(self, noisy_data, data_2d):
        return mix_data_with_condition(
            noisy_data,
            data_2d,
            mix_mode=self.mix_mode,
        )
