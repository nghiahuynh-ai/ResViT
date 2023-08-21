import os
import torch
import torch.nn as nn
from resvit.module.enc_dec import build_enc_dec
from resvit.module.bottleneck import build_bottleneck
from resvit.utils.dataset import ResViTDetectorDataset
from resvit.module.loss import BalanceBCELoss, DiceLoss
from resvit.utils.scheduler import NoamScheduler
import torchmetrics.functional as metrics
import torchvision.transforms as T
from omegaconf import DictConfig
import lightning.pytorch as pl
from resvit.utils.dataset import preprocess
from datetime import datetime


class ResViTDetector(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(ResViTDetector, self).__init__()
        
        self.cfg = cfg
        
        self.encoder, self.decoder = build_enc_dec(cfg.enc_dec, out_layer=False)

        if os.path.isfile(cfg.pretrain):
            pretrain = torch.load(cfg.pretrain, map_location=self.device)['state_dict']
            self.load_state_dict(pretrain, strict=False)
        
        self.out = nn.Sequential(
            nn.Conv2d(
                in_channels=cfg.enc_dec.init_channels,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid()
        )
        
        self.loss = {
            'ls': nn.BCELoss(),
            'lb': DiceLoss(),
            'lt': nn.L1Loss(),
        }
        
        self.train_dataset = ResViTDetectorDataset(cfg.train_dataset)
        self.validation_dataset = ResViTDetectorDataset(cfg.validation_dataset)
        self.test_dataset = ResViTDetectorDataset(cfg.test_dataset)
        
        self.optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=cfg.optim.lr,
            betas=cfg.optim.betas,
            weight_decay=cfg.optim.weight_decay,
        )
        
    def forward(self, x):
        x = self.encoder(x)
        # x = self.bottleneck(x)
        x = self.decoder(x, self.encoder.layers_outs)
        x = self.out(x)
        return x
    
    def _binarize(self, x):
        return x
    
    def training_step(self, batch, batch_idx):
        x, gt = batch
        
        x_pred = self.forward(x)
        
        loss = self.loss['ls'](x_pred, gt)
        
        log_dict = {
            "train_loss": {"value": loss, "on_step": True, "on_epoch": True, "prog_bar": True, "logger": True},
            "lr": {
                "value": self.optimizer.param_groups[0]['lr'], 
                "on_step": True, "on_epoch": True, "prog_bar": True, "logger": True
            }
        }
        self.logging(log_dict)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, gt = batch

        x_pred = self.forward(x)
        loss = self.loss['ls'](x_pred, gt)
        x_pred = ((x_pred > 0.5) * 1.0).to(self.device)
        
        precision = metrics.precision(x_pred, gt, task='binary', num_classes=2)
        recall = metrics.recall(x_pred, gt, task='binary', num_classes=2)
        f1 = metrics.f1_score(x_pred, gt, task='binary', num_classes=2)
        
        log_dict = {
            "train_loss": {"value": loss, "on_step": True, "on_epoch": True, "prog_bar": True, "logger": True},
            "lr": {
                "value": self.optimizer.param_groups[0]['lr'], 
                "on_step": True, "on_epoch": True, "prog_bar": True, "logger": True
            },
            "precision": {"value": precision, "on_step": True, "on_epoch": True, "prog_bar": True, "logger": True},
            "recall": {"value": recall, "on_step": True, "on_epoch": True, "prog_bar": True, "logger": True},
            "f1": {"value": f1, "on_step": True, "on_epoch": True, "prog_bar": True, "logger": True},
        }
        self.logging(log_dict)

        return x_pred
    
    def test_step(self, batch, batch_idx):
        x, gt = batch

        x_pred = self.forward(x)
        loss = self.loss['ls'](x_pred, gt)
        x_pred = ((x_pred > 0.5) * 1.0).to(gt.device)
        
        precision = metrics.precision(x_pred, gt, task='binary', num_classes=2)
        recall = metrics.recall(x_pred, gt, task='binary', num_classes=2)
        f1 = metrics.f1_score(x_pred, gt, task='binary', num_classes=2)
        
        log_dict = {
            "train_loss": {"value": loss, "on_step": True, "on_epoch": True, "prog_bar": True, "logger": True},
            "lr": {
                "value": self.optimizer.param_groups[0]['lr'], 
                "on_step": True, "on_epoch": True, "prog_bar": True, "logger": True
            },
            "precision": {"value": precision, "on_step": True, "on_epoch": True, "prog_bar": True, "logger": True},
            "recall": {"value": recall, "on_step": True, "on_epoch": True, "prog_bar": True, "logger": True},
            "f1": {"value": f1, "on_step": True, "on_epoch": True, "prog_bar": True, "logger": True},
        }
        self.logging(log_dict)

        return x_pred
    
    def predict(self, image=None, image_path=None, out_dir=None):
        
        image, h, w = preprocess(
            image=image,
            image_path=image_path, 
            scaling_factor=self.cfg.enc_dec.n_stages,
        )
        image = image.to(self.device)
        
        pred = self.forward(image.unsqueeze(0)).squeeze(0)
        
        pred = pred[:, :h, :w] # clip padding
        pred = (pred > 0.5) * 1.0
        transform = T.ToPILImage()
        pred = transform(pred)
        
        if out_dir is None:
            out_dir = os.getcwd()
        else:
            if not os.path.isdir(out_dir):
                raise NotADirectoryError("The given output directory does not exist!")
            
        if image_path is not None:
            name = image_path.split('/')[-1].split('.')[0] + '_bitmap_result.png'
        else:
            name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + '_bitmap_result.png'
            
        pred.save(os.path.join(out_dir, name))
        
    def logging(self, logs: dict):
        for key in logs:
            self.log(
                key,
                logs[key]['value'],
                logs[key]['on_step'],
                logs[key]['on_epoch'],
                logs[key]['prog_bar'],
                logs[key]['logger'],
            )
        
    def configure_optimizers(self):
        return {"optimizer": self.optimizer}