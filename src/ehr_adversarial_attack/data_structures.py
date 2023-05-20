import torch
from dataclasses import dataclass


@dataclass
class VariableLengthFeatures:
    features: torch.tensor
    lengths: torch.tensor