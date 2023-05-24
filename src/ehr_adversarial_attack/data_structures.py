import optuna
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class VariableLengthFeatures:
    features: torch.tensor
    lengths: torch.tensor


@dataclass
class ClassificationScores:
    accuracy: float
    roc_auc: float
    precision: float
    recall: float
    f1: float

    def __str__(self) -> str:
        return (
            f"Accuracy:\t{self.accuracy:.4f}\n"
            f"AUC:\t\t{self.roc_auc:.4f}\n"
            f"Precision:\t{self.precision:.4f}\n"
            f"Recall:\t\t{self.recall:.4f}\n"
            f"F1:\t\t\t{self.f1:.4f}"
        )


@dataclass
class EvalResults:
    loss: float
    accuracy: float
    roc_auc: float
    precision: float
    recall: float
    f1: float

    @classmethod
    def from_loss_and_scores(cls, loss: float, scores: ClassificationScores):
        return cls(
            loss=loss,
            accuracy=scores.accuracy,
            roc_auc=scores.roc_auc,
            precision=scores.precision,
            recall=scores.recall,
            f1=scores.f1,
        )

    def __str__(self) -> str:
        return (
            f"Loss:\t{self.loss}"
            f"Accuracy:\t{self.accuracy:.4f}\n"
            f"AUC:\t\t{self.roc_auc:.4f}\n"
            f"Precision:\t{self.precision:.4f}\n"
            f"Recall:\t\t{self.recall:.4f}\n"
            f"F1:\t\t\t{self.f1:.4f}"
        )
