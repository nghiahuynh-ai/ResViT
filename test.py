import time
import numpy as np
import torch
from omegaconf import OmegaConf
import torchmetrics
from resvit.module.enc_dec import DeformConv2d

a = DeformConv2d(
    in_channels=3,
    out_channels=3,
    kernel_size=3,
    stride=1,
    padding=1,
)

b = torch.rand(1, 3, 5, 5)
c = a(b)
# print(c.shape)
print(a)