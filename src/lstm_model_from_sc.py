import sklearn.metrics as skm
import torch
import torch.nn as nn
from standard_classifier import (
    StandardClassifier,
    StandardClassificationMetrics,
)


class LSTMSun2018(StandardClassifier):
    def __init__(
        self,
        device: torch.device,
        input_size: int = 48,
        lstm_hidden_size: int = 128,
        fc_hidden_size: int = 32,
        learn_rate: float = 1e-4,
        beta_1: float = 0.5,
        beta_2: float = 0.999,
    ):
        super().__init__(device=device)
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

        self.learn_rate = learn_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2

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

    # TODO create separate module with typical functions for this
    @staticmethod
    def get_predicted_class(y_hat: torch.tensor) -> torch.tensor:
        return torch.argmax(y_hat, dim=1)

    # TODO create separate module with typical functions for this
    @staticmethod
    def get_classification_metrics(
        y_score: torch.tensor, y_pred: torch.tensor, y_true: torch.tensor
    ) -> StandardClassificationMetrics:
        y_true_one_hot = torch.nn.functional.one_hot(y_true)
        y_score = y_score.detach().numpy()
        y_pred = y_pred.detach().numpy()
        y_true = y_true.detach().numpy()

        return StandardClassificationMetrics(
            accuracy=skm.accuracy_score(y_true=y_true, y_pred=y_pred),
            roc_auc=skm.roc_auc_score(y_true=y_true_one_hot, y_score=y_score),
            precision=skm.precision_score(y_true=y_true, y_pred=y_pred),
            recall=skm.recall_score(y_true=y_true, y_pred=y_pred),
            f1=skm.f1_score(y_true=y_true, y_pred=y_pred),
        )
