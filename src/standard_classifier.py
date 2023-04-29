import torch
import torch.nn as nn
import torch.utils.data as ud
from abc import ABC, abstractmethod
from dataclasses import dataclass


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


class StandardClassifier(ABC, nn.Module):
    def __init__(self, device: torch.device):
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
        self,
        train_loader: ud.DataLoader,
        num_epochs: int,
        loss_log: list[float] = None,
    ):
        self.train()
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
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
            if loss_log is not None:
                loss_log.append(epoch_loss)

    def evaluate_model(
        self, test_loader: ud.DataLoader, metrics_log: list[dataclass] = None
    ):
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
        print(f"Predictive performance on test data:\n{metrics}\n")

        if metrics_log is not None:
            metrics_log.append(metrics)
