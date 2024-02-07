import torch
import torch.nn as nn


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, input, target):
        return torch.nn.functional.l1_loss(input, target)