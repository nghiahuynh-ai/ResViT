import math
import cv2
import torch
import numpy as np
import PIL.Image as Image
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig
from resvit.utils.find_files import find_files_by_ext

class ResViTSSLCollate:
    
    def __init__(self, scaling_factor, patch_size):
        self.total_downsample_factor = 2**scaling_factor * patch_size
        
    def __call__(self, batch):
        
        max_h, max_w = 0, 0
        for sample in batch:
            _, h, w = sample.shape
            if h > max_h:
                max_h = h
            if w > max_w:
                max_w = w

        max_h = math.ceil(max_h / self.total_downsample_factor) * self.total_downsample_factor
        max_w = math.ceil(max_w / self.total_downsample_factor) * self.total_downsample_factor
        
        samples, sample_size = [], []
        for sample in batch:
            _, h, w = sample.shape
            pad = (0, max_w - w, 0, max_h - h)
            sample = F.pad(sample, pad, "constant", 0)
            samples.append(sample)
            sample_size.append(torch.tensor([h, w], dtype=torch.long))
        samples = torch.stack(samples)
        sample_size = torch.stack(sample_size)
        return samples, sample_size
    
    
class ResViTDetectorCollate:
    
    def __init__(self):
        pass
    
    def __call__(self, batch):
        pass
    

class ResViTSSLDataset(Dataset):

    def __init__(self, cfg: DictConfig):
        
        self.root_dir = cfg.root_dir
        self.samples = find_files_by_ext(cfg.root_dir, cfg.extensions)
        self.transform = transforms.ToTensor()
            
        collate = ResViTSSLCollate(cfg.scaling_factor, cfg.patch_size)
        self.loader = DataLoader(
            self, 
            batch_size=cfg.batch_size, 
            shuffle=cfg.shuffle,
            num_workers=cfg.num_workers,
            collate_fn=collate,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        img_name = self.samples[idx]
        image = Image.open(img_name)
        image = np.array(image.convert('RGB'))
        image = cv2.normalize(image, None, alpha=0,beta=255, norm_type=cv2.NORM_MINMAX)
        image = self.transform(image)

        return image
    

class ResViTDetectorDataset(Dataset):
    
    def __init__(self, cfg):
        pass
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass
    