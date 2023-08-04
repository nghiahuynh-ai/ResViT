import time
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

# from resvit.utils.find_files import find_files_by_ext

# files = find_files_by_ext('resvit', ['.py', '.pyc'])
# print(files)