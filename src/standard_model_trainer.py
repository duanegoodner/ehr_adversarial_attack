import sklearn.metrics as skm
import torch.nn as nn
import torch.optim
import torch.utils.data as ud
from dataclasses import dataclass
import standard_trainable_classifier as stc

from x19_mort_dataset import X19MortalityDataset
from lstm_model_stc import LSTMSun2018


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
        model: stc.StandardTrainableClassifier,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    @staticmethod
    def interpret_output(model_output: torch.tensor) -> torch.tensor:
        return torch.argmax(input=model_output, dim=1)

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

    def train_model(self, train_loader: ud.DataLoader, num_epochs: int):
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for num_batches, (x, y) in enumerate(train_loader):
                # y = y.long()
                x, y = x.to(self.model.model_device), y.to(self.model.model_device)
                self.optimizer.zero_grad()
                y_hat = self.model(x).squeeze()
                loss = self.loss_fn(y_hat, y)
                loss.backward()
                loss.to("cpu")
                x.to("cpu")
                y.to("cpu")
                self.optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / (num_batches + 1)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    def evaluate_model(self, test_loader: ud.DataLoader):
        self.model.eval()
        all_y_true = torch.LongTensor()
        all_y_pred = torch.LongTensor()
        all_y_score = torch.FloatTensor()
        for x, y in test_loader:
            x, y = x.to(self.model.model_device), y.to(self.model.model_device)
            y_hat = self.model(x)
            y_pred = self.interpret_output(model_output=y_hat)
            all_y_true = torch.cat((all_y_true, y.to("cpu")), dim=0)
            all_y_pred = torch.cat((all_y_pred, y_pred.to("cpu")), dim=0)
            all_y_score = torch.cat((all_y_score, y_hat.to("cpu")), dim=0)
        metrics = self.calculate_performance_metrics(
            y_score=all_y_score, y_pred=all_y_pred, y_true=all_y_true
        )
        print(f"Predictive performance on test data:\n{metrics}\n")


if __name__ == "__main__":
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    dataset = X19MortalityDataset()
    data_loader = ud.DataLoader(dataset=dataset, batch_size=128, shuffle=True)
    cur_model = LSTMSun2018(model_device=cur_device)
    trainer = StandardModelTrainer(
        model=cur_model,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(
            params=cur_model.parameters(), lr=1e-4, betas=(0.5, 0.999)
        ),
    )

    trainer.train_model(train_loader=data_loader, num_epochs=5)
    trainer.evaluate_model(test_loader=data_loader)
