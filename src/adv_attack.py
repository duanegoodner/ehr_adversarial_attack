import numpy as np
import torch
import torch.utils.data as ud
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from lstm_model_stc import LSTMSun2018
from standard_model_evaluator import (
    StandardInferenceResults,
    StandardModelInferrer,
)
from standard_trainable_classifier import StandardTrainableClassifier
from x19_mort_dataset import X19MortalityDataset


class AdversarialAttacker:
    def __init__(
        self, model: StandardTrainableClassifier, dataset: ud.Dataset
    ):
        self.model = model
        self.dataset = dataset
        self.inferrer = StandardModelInferrer(model=model, dataset=dataset)

    def get_correct_predictions_dataset(self) -> ud.Dataset:
        inference_results = self.inferrer.infer()
        return ud.Subset(
            dataset=self.dataset,
            indices=inference_results.correct_prediction_indices,
        )


if __name__ == "__main__":
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    model = LSTMSun2018(model_device=cur_device)
    checkpoint_path = Path(
        "/home/duane/dproj/UIUC-DLH/project/ehr_adversarial_attack/data"
        "/training_results/2023-04-30_18:49:09.556432.tar"
    )
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset = X19MortalityDataset()
    adv_attacker = AdversarialAttacker(model=model, dataset=dataset)
