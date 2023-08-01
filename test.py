import time
import torch
import numpy as np
from torchsummary import summary
from skimage import io, transform
from unet.enc_dec import build_enc_dec
from unet.bottleneck import build_bottleneck
from omegaconf import DictConfig, OmegaConf, open_dict
from unet.dataset import UNetReconstructDataset


config = OmegaConf.load('config/cfg.yaml')

enc, dec = build_enc_dec(config.enc_dec)
bottleneck = build_bottleneck(config.bottleneck)

print('==========================================================================================================================')
print(enc)
print('==========================================================================================================================')
print(bottleneck)
print('==========================================================================================================================')
print(dec)

x = torch.rand(1, 3, 1920, 1152)
# x = torch.rand(4, 3, 2688, 1536)
print(x.shape)
s = time.time()
x = enc(x)
print(x.shape)
x = bottleneck(x)
print(x.shape)
x = dec(x, enc.layers_outs)
e = time.time()
print(x.shape)
print(e-s)


# image = io.imread('examples/data/4e0df6a064e28dbcd4f3.jpg')
# image = np.pad(image, (8, 10), 'constant', constant_values=0)
# print(image.shape, type(image))

# image = torch.ones_like(torch.empty(1, 5, 7))
# image = io.imread('examples/data/4e0df6a064e28dbcd4f3.jpg')
# print(image.shape)
# out = torch.nn.functional.pad(torch.as_tensor(image, dtype=torch.float32), (0, 2, 0, 1), "constant", 0)
# print(out.shape)

# dataset = UNetReconstructDataset(config.train_dataset)
# datasetloader = dataset.loader
# for batch in datasetloader:
#     print(batch.shape)