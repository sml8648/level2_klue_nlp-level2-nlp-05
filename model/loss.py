import torch.nn.functional as F
from torch import nn
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)

def crossentropy_loss(output, target):
    return nn.CrossEntropyLoss(output, target)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss()
        ce_loss = ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
        
loss_config = {
    "nll": nll_loss,
    "crossentropy": crossentropy_loss,
}