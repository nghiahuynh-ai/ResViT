import torch
import torch.nn as nn
import torchmetrics.functional as f


class BalanceBCELoss(nn.Module):
    
    def __init__(self):
        super(BalanceBCELoss, self).__init__()
        self.loss = nn.BCELoss()
        
    def __call__(self, gt, prob, k):
        pass
    
    
class DiceLoss(nn.Module):
    
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, preds, targets):
        return 1 - f.dice(preds, targets)
        