import random
import torch
import torch.nn as nn
from omegaconf import DictConfig


class RectangleMasking(nn.Module):
    
    def __init__(self, cfg: DictConfig):
        super(RectangleMasking, self).__init__()
        
        self.num_mask = cfg.num_mask
        self.height = cfg.height
        self.width = cfg.width
        self.mask_value = cfg.mask_value
        
    @torch.no_grad()
    def forward(self, x):
        b, _, h, w = x.shape

        for idx in range(b):
            for _ in range(self.num_mask):
                h_start = random.randint(0, h - int(h * self.height))
                w_start = random.randint(0, w - int(h * self.width))
                h_offet = random.randint(0, int(h * self.height))
                w_offet = random.randint(0, int(h * self.width))
                x[idx, :, h_start : h_start + h_offet, w_start : w_start + w_offet] = self.mask_value
                
        return x
    
    
class PixelMasking(nn.Module):
    
    def __init__(self, cfg: DictConfig):
        super(PixelMasking, self).__init__()
        
        self.mask_ratio = cfg.ratio
        self.mask_value = cfg.mask_value
        
    @torch.no_grad()
    def forward(self, x):
        prob = -1.0 * torch.rand(x.shape) + 1.0
        mask = prob > self.mask_ratio
        x = x * mask.to(x.device)
        return x