import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as ud
from dataclasses import dataclass
from standard_trainable_classifier import StandardAttackableClassifier


@dataclass
class StandardInferenceResults:
    y_pred: torch.tensor
    y_score: torch.tensor
    y_true: torch.tensor = None

    @property
    def correct_prediction_indices(self) -> np.ndarray:
        return np.where(self.y_pred == self.y_true)


class StandardModelInferrer:
    def __init__(
        self,
        model: nn.Module,
        dataset: ud.Dataset,
        batch_size: int = 64,
    ):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.data_loader = ud.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False
        )

    # extract this from evaluate function so it can be easily overridden
    @staticmethod
    def interpret_output(model_output: torch.tensor) -> torch.tensor:
        return torch.argmax(input=model_output, dim=1)

    @torch.no_grad()
    def infer(self):
        self.model.eval()
        all_y_true = torch.LongTensor()
        all_y_pred = torch.LongTensor()
        all_y_score = torch.FloatTensor()
        for x, y in self.data_loader:
            x, y = x.to(self.model.model_device), y.to(self.model.model_device)
            y_hat = self.model(x)
            y_pred = self.interpret_output(model_output=y_hat)
            all_y_true = torch.cat((all_y_true, y.to("cpu")), dim=0)
            all_y_pred = torch.cat((all_y_pred, y_pred.to("cpu")), dim=0)
            all_y_score = torch.cat((all_y_score, y_hat.to("cpu")), dim=0)
        return StandardInferenceResults(
            y_pred=all_y_pred, y_score=all_y_score, y_true=all_y_true
        )

    @torch.no_grad()
    def get_logits(self, x: torch.tensor):
        self.model.eval()
        logits = self.model.logit_output(x=x)
        return logits
