import sklearn.metrics as skm
import torch
import torch.nn as nn
import torch.utils.data as ud
from typing import NamedTuple


class ClassificationMetrics(NamedTuple):
    accuracy: float
    roc_auc: float
    precision: float
    recall: float
    f1: float


def get_classification_metrics(
    y_score: torch.FloatTensor,
    y_pred: torch.LongTensor,
    y_true: torch.LongTensor,
) -> ClassificationMetrics:
    return ClassificationMetrics(
        accuracy=skm.accuracy_score(y_true=y_true, y_pred=y_pred),
        roc_auc=skm.roc_auc_score(y_true=y_true, y_score=y_score),
        precision=skm.precision_score(y_true=y_true, y_pred=y_pred),
        recall=skm.recall_score(y_true=y_true, y_pred=y_pred),
        f1=skm.f1_score(y_true=y_true, y_pred=y_pred),
    )


class BinaryBidirectionalLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        lstm_hidden_size: int,
        fc_hidden_size: int,
        learn_rate: float = 1e-4,
        beta_1: float = 0.5,
        beta_2: float = 0.999,
    ):
        super(BinaryBidirectionalLSTM, self).__init__()
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=0.5)
        self.fc_1 = nn.Linear(
            in_features=2 * lstm_hidden_size, out_features=fc_hidden_size
        )
        self.act_1 = nn.ReLU()
        self.fc_2 = nn.Linear(in_features=fc_hidden_size, out_features=1)
        self.act_2 = nn.Sigmoid()

        self.loss_fn = nn.BCELoss()
        self.optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=learn_rate,
            betas=(beta_1, beta_2),
        )

    def forward(self, x: torch.tensor):
        h_0 = torch.zeros(2, x.size(0), self.lstm_hidden_size)
        c_0 = torch.zeros(2, x.size(0), self.lstm_hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        lstm_out = nn.Tanh(lstm_out)
        lstm_out = self.dropout(lstm_out)
        fc_1_out = self.fc_1(lstm_out[:, -1, :])
        fc_1_out = self.act_1(fc_1_out)
        fc_2_out = self.fc_2(fc_1_out)
        out = self.act_2(fc_2_out)
        return out

    def train_model(self, train_loader: ud.DataLoader, num_epochs: int):
        self.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (x, y) in enumerate(train_loader):
                self.optimizer.zero_grad()
                y_hat = self(x)
                loss = self.loss_fn(y_hat, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(
                "Epoch [%d/%d], Loss: %.4f"
                % (epoch + 1, num_epochs, running_loss / (i + 1))
            )

    def evaluate_model(self, test_loader: ud.DataLoader):
        self.eval()
        all_y_true = torch.LongTensor()
        all_y_pred = torch.LongTensor()
        all_y_score = torch.FloatTensor()
        for x, y in test_loader:
            y_hat = self(x)
            y_pred = (y_hat >= 0.5).type(torch.long)
            all_y_true = torch.cat((all_y_true, y.to("cpu")), dim=0)
            all_y_pred = torch.cat((all_y_pred, y_pred.to("cpu")), dim=0)
            all_y_score = torch.cat((all_y_score, y_hat.to("cpu")), dim=0)

        metrics = get_classification_metrics(
            y_score=all_y_score, y_pred=all_y_pred, y_true=all_y_true
        )

        print(
            f"acc: {metrics.accuracy:.3f}\n"
            f"auc: {metrics.roc_auc:.3f}\n"
            f"precision: {metrics.precision:.3f}\n"
            f"recall: {metrics.recall:.3f}\n"
            f"f1: {metrics.f1:.3f}"
        )
