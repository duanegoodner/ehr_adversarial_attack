import sklearn.metrics as skm
import torch.nn as nn
import torch.optim
import torch.utils.data as ud
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


# TODO Separate Trainer and Evaluator in to two classes


class ModuleWithDevice(nn.Module):
    def __init__(self, device: torch.device):
        super(ModuleWithDevice, self).__init__()
        self.device = device
        self.to(device)


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


class StandardModelTrainer:
    def __init__(
        self,
        device: torch.device,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: ud.DataLoader,
        test_loader: ud.DataLoader,
        checkpoint_dir: Path = None,
        epoch_start_count: int = 0,
    ):
        self.device = device
        self.model = model
        self.model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.checkpoint_dir = checkpoint_dir
        self.total_epochs = epoch_start_count

    @staticmethod
    def calculate_performance_metrics(
        y_score: torch.tensor, y_pred: torch.tensor, y_true: torch.tensor
    ) -> StandardClassificationMetrics:
        y_true_one_hot = torch.nn.functional.one_hot(y_true)
        y_score_np = y_score.detach().numpy()
        y_pred_np = y_pred.detach().numpy()
        y_true_np = y_true.detach().numpy()

        return StandardClassificationMetrics(
            accuracy=skm.accuracy_score(y_true=y_true_np, y_pred=y_pred_np),
            roc_auc=skm.roc_auc_score(
                y_true=y_true_one_hot, y_score=y_score_np
            ),
            precision=skm.precision_score(y_true=y_true_np, y_pred=y_pred_np),
            recall=skm.recall_score(y_true=y_true_np, y_pred=y_pred_np),
            f1=skm.f1_score(y_true=y_true_np, y_pred=y_pred_np),
        )

    def save_checkpoint(
        self,
        loss_log: list[float],
        metrics: StandardClassificationMetrics,
    ) -> Path:
        filename = f"{datetime.now()}.tar".replace(" ", "_")
        output_path = self.checkpoint_dir / filename
        output_object = {
            "epoch_num": self.total_epochs,
            "loss_log": loss_log,
            "metrics": metrics,
            "state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(obj=output_object, f=output_path)
        return output_path

    def train_model(
        self,
        num_epochs: int,
    ):
        self.model.train()

        loss_log = []
        for epoch in range(num_epochs):
            running_loss = 0.0
            for num_batches, (inputs, y) in enumerate(self.train_loader):
                inputs.features, y = (
                    inputs.features.to(self.device),
                    y.to(self.device),
                )
                self.optimizer.zero_grad()
                y_hat = self.model(inputs).squeeze()
                loss = self.loss_fn(y_hat, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / (num_batches + 1)
            loss_log.append(epoch_loss)
            # TODO move reporting work to separate method(s)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        self.total_epochs += num_epochs
        return loss_log

    @torch.no_grad()
    def evaluate_model(self):
        self.model.eval()
        all_y_true = torch.LongTensor()
        all_y_pred = torch.LongTensor()
        all_y_score = torch.FloatTensor()
        for num_batches, (inputs, y) in enumerate(self.train_loader):
            inputs.features, y = inputs.features.to(self.device), y.to(self.device)
            y_hat = self.model(inputs)
            y_pred = torch.argmax(input=y_hat, dim=1)
            all_y_true = torch.cat((all_y_true, y.to("cpu")), dim=0)
            all_y_pred = torch.cat((all_y_pred, y_pred.to("cpu")), dim=0)
            all_y_score = torch.cat((all_y_score, y_hat.to("cpu")), dim=0)
        metrics = self.calculate_performance_metrics(
            y_score=all_y_score, y_pred=all_y_pred, y_true=all_y_true
        )
        print(f"Predictive performance on test data:\n{metrics}\n")
        return metrics

    def run_train_eval_cycles(self, epochs_per_cycle: int, max_num_cycles: int, save_checkpoints: bool=True):
        for cycle_num in range(max_num_cycles):
            loss_log = self.train_model(num_epochs=epochs_per_cycle)
            eval_metrics = self.evaluate_model()
            if save_checkpoints:
                self.save_checkpoint(loss_log=loss_log, metrics=eval_metrics)



