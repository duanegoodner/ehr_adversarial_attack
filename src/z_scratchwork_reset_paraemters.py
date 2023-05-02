import torch.nn as nn
import torch


class RegularizedFC(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    # def reset_parameters(self):
    #     nn.init.zeros_(self.bias)

    def forward(self, x):
        return x + self.bias

    def l1_loss(self, alpha):
        return alpha * torch.norm(self.bias, p=1)

