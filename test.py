import time
import numpy as np
import torch
from omegaconf import OmegaConf
from resvit.model.resvit_ssl import ResViTSSL

# config = OmegaConf.load('examples/config/ssl.yaml')

# model = ResViTSSL(config)
# x = torch.rand(1, 3, 1920, 1152)
# s = time.time()
# x = model(x)
# x = model.post_process(x, torch.tensor([[1920, 1152]]))
# e = time.time()
# print(e-s)

from resvit.utils.gen_label import gen_label

a = np.random.rand(5, 5, 1)
polygons = [[[1,2],[1,4], [2,2], [2,4]], [[3,3],[3,4], [4,3], [4,4]]]
b = gen_label(a, np.array(polygons))

print(b)