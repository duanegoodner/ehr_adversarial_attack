import sklearn.metrics as skm
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable


def choose_max_val(x: torch.tensor) -> torch.tensor:
    return torch.argmax(input=x, dim=1)


@dataclass
class StandardClassificationMetrics:
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


def calc_standard_metrics(
    y_score: torch.tensor, y_pred: torch.tensor, y_true: torch.tensor
) -> StandardClassificationMetrics:
    y_true_one_hot = torch.nn.functional.one_hot(y_true)
    y_score_np = y_score.detach().numpy()
    y_pred_np = y_pred.detach().numpy()
    y_true_np = y_true.detach().numpy()

    return StandardClassificationMetrics(
        accuracy=skm.accuracy_score(y_true=y_true_np, y_pred=y_pred_np),
        roc_auc=skm.roc_auc_score(y_true=y_true_one_hot, y_score=y_score_np),
        precision=skm.precision_score(y_true=y_true_np, y_pred=y_pred_np),
        recall=skm.recall_score(y_true=y_true_np, y_pred=y_pred_np),
        f1=skm.f1_score(y_true=y_true_np, y_pred=y_pred_np),
    )


class StandardTrainableClassifier(nn.Module, ABC):
    def __init__(
        self,
        model_device: torch.device,
        metrics_calculator: Callable[
            [torch.tensor, torch.tensor, torch.tensor], dataclass
        ] = calc_standard_metrics,
        output_interpreter: Callable[
            [torch.tensor], torch.tensor
        ] = choose_max_val,
    ):
        super(StandardTrainableClassifier, self).__init__()
        self.model_device = model_device
        self.output_interpreter = output_interpreter
        self.metrics_calculator = metrics_calculator
        self.to(device=model_device)

    @abstractmethod
    def forward(self, x: torch.tensor) -> torch.tensor:
        pass


