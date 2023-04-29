import torch
import torch.nn as nn
import torch.utils.data as ud
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple


@dataclass
class StandardClassificationMetrics:
    accuracy: float
    roc_auc: float
    precision: float
    recall: float
    f1: float


class StandardClassifier(ABC, nn.Module):
    def __init__(
        self,
        device: torch.device
    ):
        super(StandardClassifier, self).__init__()
        self._device = device
        self._loss_fn = None
        self._optimizer = None
        self.to(self._device)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def loss_fn(self) -> nn.Module | None:
        return self._loss_fn

    def set_loss_fn(self, loss: nn.Module):
        self._loss_fn = loss
        return self

    @property
    def optimizer(self) -> torch.optim.Optimizer | None:
        return self._optimizer

    def set_optimizer(self, opt: torch.optim.Optimizer):
        self._optimizer = opt
        return self

    @abstractmethod
    def forward(self, x: torch.tensor) -> torch.tensor:
        pass

    @staticmethod
    @abstractmethod
    def get_predicted_class(y_hat: torch.tensor) -> torch.tensor:
        pass

    @staticmethod
    @abstractmethod
    def get_classification_metrics(
        y_score: torch.tensor, y_pred: torch.tensor, y_true: torch.tensor
    ) -> dataclass:
        pass

    def train_model(
        self, train_loader: ud.DataLoader, num_epochs: int
    ) -> list[float]:
        self.train()
        each_epoch_loss = []
        for epoch in range(num_epochs):
            running_loss = 0.0
            for num_batches, (x, y) in enumerate(train_loader):
                # y = y.long()
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self(x).squeeze()
                loss = self.loss_fn(y_hat, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / (num_batches + 1)
            print(
                "Epoch [%d/%d], Loss: %.4f"
                % (epoch + 1, num_epochs, epoch_loss)
            )
            each_epoch_loss.append(epoch_loss)
        return each_epoch_loss

    def evaluate_model(self, test_loader: ud.DataLoader) -> dataclass:
        self.eval()
        all_y_true = torch.LongTensor()
        all_y_pred = torch.LongTensor()
        all_y_score = torch.FloatTensor()
        for x, y in test_loader:
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self(x)
            y_pred = self.get_predicted_class(y_hat=y_hat)
            all_y_true = torch.cat((all_y_true, y.to("cpu")), dim=0)
            all_y_pred = torch.cat((all_y_pred, y_pred.to("cpu")), dim=0)
            all_y_score = torch.cat((all_y_score, y_hat.to("cpu")), dim=0)

        metrics = self.get_classification_metrics(
            y_score=all_y_score, y_pred=all_y_pred, y_true=all_y_true
        )

        return metrics
