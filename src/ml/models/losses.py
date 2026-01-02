import numpy as np
import torch
from torch.nn import functional as F

class ClassBalancedLoss(torch.nn.Module):
    def __init__(self, gamma, n_samples_per_classes, loss_type='sigmoid'):
        super(ClassBalancedLoss, self).__init__()
        self.gamma = gamma
        self.effective_nums = 1.0 - np.power(gamma, n_samples_per_classes)
        self.n_classes = len(n_samples_per_classes)
        self.loss_type = loss_type
    
    def forward(self, logits, labels):
        labels = labels.to(torch.int64)
        labels_one_hot = F.one_hot(labels, self.n_classes).float()
        weights = (1.0 - self.gamma) / np.array(self.effective_nums)
        weights = weights / np.sum(weights) * self.n_classes
        weights = torch.tensor(weights, device=logits.device).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1,self.n_classes)
        if self.loss_type == "sigmoid":
            loss = F.binary_cross_entropy_with_logits(input=logits,target=labels_one_hot,weight=weights)
        elif self.loss_type == "sofmax":
            pred = logits.softmax(dim=1)
            loss = F.binary_cross_entropy(input=pred,target=labels_one_hot,weight=weights)
        else:
            raise ValueError("loss_type must be sigmoid or softmax")
        return loss

class DAMLoss(torch.nn.Module):
    def __init__(self, class_counts, scale=1.0):
        super().__init__()
        self.margins = scale / torch.sqrt(class_counts.clone().detach().float())

    def forward(self, logits, targets):
        targets = targets.to(torch.long)  # Ensure labels are integer indices
        margins = self.margins.to(logits.device)
        adjusted_logits = logits - margins[targets]
        return F.cross_entropy(adjusted_logits, targets.to(torch.float) )
