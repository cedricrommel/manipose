from logging import warn
import math

from torch import nn
from mup import init


def mu_init_params(model):
    for name, param in model.named_parameters():
        try:
            print(name, param.shape)
            if name.endswith("weight"):
                init.kaiming_uniform_(param, a=math.sqrt(5))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(param)
            elif name.endswith("bias"):
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(param, -bound, bound)
            else:
                warn(f"Could not init {name}")
        except AssertionError and ValueError:
            warn(f"Could not init {name}")
