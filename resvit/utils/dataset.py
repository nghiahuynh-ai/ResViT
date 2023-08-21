import json
import math
import os
import cv2
import torch
import numpy as np
import PIL.Image as Image
from omegaconf import DictConfig
from torchvision import transforms
from torch.nn import functional as F
from resvit.utils.gen_label import gen_label
from torch.utils.data import Dataset, DataLoader
from resvit.utils.find_files import find_files_by_ext
from resvit.utils.augmentation import ImgAugTransform


def preprocess(image=None, image_path=None, scaling_factor=6, patch_size=3):
    
    is_array = image is not None
    is_path = image_path is not None
    if (is_array ^ is_path) is False:
        raise ValueError("Arguments ``image`` and ``image_path`` must be mutually exclusive")
    
    if is_array:
        assert isinstance(image, np.ndarray)
        
    if is_path:
        image = Image.open(image_path)
        image = np.array(image.convert('RGB'))
        
    image = cv2.normalize(image, None, alpha=0,beta=255, norm_type=cv2.NORM_MINMAX)
    transform = transforms.ToTensor()
    image = transform(image)
    
    _, h, w = image.shape   
    total_downsample_factor = 2**scaling_factor * patch_size
    max_h = math.ceil(h / total_downsample_factor) * total_downsample_factor
    max_w = math.ceil(w / total_downsample_factor) * total_downsample_factor
    pad = (0, max_w - w, 0, max_h - h)
    image = F.pad(image, pad, "constant", 0)
    
    return image, h, w
    

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
    
    def __init__(self, scaling_factor, patch_size, resize_dim=-1, augment=False):
        self.total_downsample_factor = 2**scaling_factor * patch_size
        self.transform = transforms.ToTensor()
        self.resize_dim = resize_dim
        if augment:
            self.augmentor = ImgAugTransform()
        else:
            self.augmentor = None
        
    def __call__(self, batch):
        
        max_h, max_w = 0, 0
        samples, groundtruths = [], []
        
        for sample, polygons in batch:
            image = Image.open(sample)
            image = np.array(image.convert('RGB'))
            
            # resize
            if self.resize_dim > 0:
                h, w, _ = image.shape
                dim_min = min(h, w)
                if h < w:
                    dim = (self.resize_dim, (w // dim_min)*w)
                else:
                    dim = ((h // dim_min)*h, self.resize_dim)
                image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
            
            # normalize
            image = cv2.normalize(image, None, alpha=0,beta=255, norm_type=cv2.NORM_MINMAX)
            
            # augment 
            if self.augmentor is not None:
                image = self.augmentor(image)
                
            gt = gen_label(image, np.array(polygons))
            
            samples.append(image)
            groundtruths.append(gt)
            
            h, w, _ = image.shape
            if h > max_h:
                max_h = h
            if w > max_w:
                max_w = w

        max_h = math.ceil(max_h / self.total_downsample_factor) * self.total_downsample_factor
        max_w = math.ceil(max_w / self.total_downsample_factor) * self.total_downsample_factor
        
        for idx in range(len(samples)):
            samples[idx] = self.transform(samples[idx])
            groundtruths[idx] = self.transform(groundtruths[idx])
            
            _, h, w = samples[idx].shape
            pad = (0, max_w - w, 0, max_h - h)

            samples[idx] = F.pad(samples[idx], pad, "constant", 0)
            groundtruths[idx] = F.pad(groundtruths[idx], pad, "constant", 0)

        samples = torch.stack(samples)
        groundtruths = torch.stack(groundtruths)
        
        return samples, groundtruths
    

class ResViTSSLDataset(Dataset):

    def __init__(self, cfg: DictConfig):
        
        self.root_dir = cfg.root_dir
        self.samples = find_files_by_ext(cfg.root_dir, cfg.extensions, acc=[])
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
    
    def __init__(self, cfg: DictConfig):
        if not os.path.isfile(cfg.manifest_path):
            raise FileNotFoundError
        else:
            samples = []
            with open(cfg.manifest_path, 'r') as f:
                for line in f:
                    line = json.loads(line)
                    if os.path.isfile(line['path']):
                        samples.append((line['path'], line['box']))
            self.samples = samples

        collate = ResViTDetectorCollate(
            scaling_factor=cfg.scaling_factor, 
            patch_size=cfg.patch_size, 
            augment=cfg.augment,
        )
        
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
        return self.samples[idx]
    