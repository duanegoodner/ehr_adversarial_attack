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
    y_true_one_hot: torch.LongTensor,
) -> ClassificationMetrics:
    return ClassificationMetrics(
        accuracy=skm.accuracy_score(y_true=y_true, y_pred=y_pred),
        roc_auc=skm.roc_auc_score(y_true=y_true_one_hot, y_score=y_score),
        precision=skm.precision_score(y_true=y_true, y_pred=y_pred),
        recall=skm.recall_score(y_true=y_true, y_pred=y_pred),
        f1=skm.f1_score(y_true=y_true, y_pred=y_pred),
    )


class BinaryBidirectionalLSTM(nn.Module):
    def __init__(
        self,
        device: torch.device,
        input_size: int,
        lstm_hidden_size: int,
        fc_hidden_size: int,
        learn_rate: float = 1e-4,
        beta_1: float = 0.5,
        beta_2: float = 0.999,
    ):
        super(BinaryBidirectionalLSTM, self).__init__()
        self.device = device
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.act_lstm = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)
        self.fc_1 = nn.Linear(
            in_features=2 * lstm_hidden_size, out_features=fc_hidden_size
        )
        self.act_1 = nn.ReLU()
        self.fc_2 = nn.Linear(in_features=fc_hidden_size, out_features=2)
        # self.act_2 = nn.Sigmoid()
        self.act_2 = nn.Softmax(dim=1)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=learn_rate,
            betas=(beta_1, beta_2),
        )

    def forward(self, x: torch.tensor):
        h_0 = torch.zeros(2, x.size(0), self.lstm_hidden_size).to(self.device)
        c_0 = torch.zeros(2, x.size(0), self.lstm_hidden_size).to(self.device)
        lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        lstm_out = self.act_lstm(lstm_out)
        lstm_out = self.dropout(lstm_out)
        fc_1_out = self.fc_1(lstm_out[:, -1, :])
        fc_1_out = self.act_1(fc_1_out)
        fc_2_out = self.fc_2(fc_1_out)
        out = self.act_2(fc_2_out)
        # out = torch.softmax(fc_2_out, dim=1)
        return out

    def train_model(
        self,
        train_loader: ud.DataLoader,
        num_epochs: int,
    ):
        self.train()  # should this be set here or by CrossValidator???
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (x, y) in enumerate(train_loader):
                y = y.long()
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self(x).squeeze()
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
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self(x)
            # y_pred = (y_hat >= 0.5).type(torch.long)
            y_pred = torch.argmax(y_hat, dim=1)
            all_y_true = torch.cat((all_y_true, y.to("cpu")), dim=0)
            all_y_pred = torch.cat((all_y_pred, y_pred.to("cpu")), dim=0)
            all_y_score = torch.cat((all_y_score, y_hat.to("cpu")), dim=0)

        metrics = get_classification_metrics(
            y_score=all_y_score.detach().numpy(),
            y_pred=all_y_pred.detach().numpy(),
            y_true=all_y_true.detach().numpy(),
            y_true_one_hot=torch.nn.functional.one_hot(all_y_true).detach().numpy()
        )

        print(
            f"acc: {metrics.accuracy:.3f}\n"
            f"auc: {metrics.roc_auc:.3f}\n"
            f"precision: {metrics.precision:.3f}\n"
            f"recall: {metrics.recall:.3f}\n"
            f"f1: {metrics.f1:.3f}"
        )
