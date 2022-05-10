import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    def __init__(self, lam=1):
        super(MultiTaskLoss, self).__init__()
        self.lam = lam
        self.cls_loss = nn.CrossEntropyLoss()
        self.reg_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, p, u, t, v):
        cls_loss = self.cls_loss(p, u)
        reg_loss = self.reg_loss(t, v).mean(dim=1)
        return cls_loss + self.lam * ((u > 0).float() * reg_loss).mean(), cls_loss, self.lam * ((u > 0).float() * reg_loss).mean()
