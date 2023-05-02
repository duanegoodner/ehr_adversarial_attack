import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as ud
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path
from lstm_model_stc import LSTMSun2018
from standard_model_inferrer import (
    StandardInferenceResults,
    StandardModelInferrer,
)
from standard_trainable_classifier import StandardAttackableClassifier
from x19_mort_dataset import X19MortalityDataset


class AdversarialLoss(nn.Module):
    def __init__(self, kappa=0):
        super(AdversarialLoss, self).__init__()
        self.kappa = kappa

    def forward(self, logits: torch.tensor, orig_label: int) -> torch.tensor:
        # return max(
        #     logits[orig_label] - logits[int(not orig_label)], self.kappa
        # )
        return logits[orig_label] - logits[int(not orig_label)]


class AdversarialAttacker(nn.Module):
    def __init__(
        self,
        target_model: StandardAttackableClassifier,
        target_dataset: ud.Dataset,
        device: torch.device,
    ):
        super(AdversarialAttacker, self).__init__()
        self.target_dataset = target_dataset
        self.orig_feature = target_dataset[0][0]
        self.orig_feature = self.orig_feature.to(device)
        self.orig_label = target_dataset[0][1]
        self.orig_label = self.orig_label.to(device)
        self.data_loader = ud.DataLoader(dataset=target_dataset, batch_size=1)
        self.inferrer = StandardModelInferrer(
            model=target_model, dataset=target_dataset, batch_size=1
        )
        self.perturbed_feature = nn.Parameter(
           self.target_dataset[0][0][None, :, :], requires_grad=True
        )
        self.loss_fn = AdversarialLoss()
        self.optimizer = torch.optim.SGD(params=self.parameters(), lr=0.001)
        self.l1_beta = 0.001
        self.device = device
        self.to(device)

    def forward(self) -> torch.tensor:
        return self.inferrer.get_logits(x=self.perturbed_feature)

    def l1_loss(self):
        return self.l1_beta * torch.norm(
            self.perturbed_feature - self.orig_feature
        )

    def run_attack(self, num_epochs: int):
        self.train()
        # self.perturbed_feature.copy_(self.target_dataset[0][0])

        for epoch in range(num_epochs):
            # for num_batches, (x, y) in enumerate(self.data_loader):
            #     x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = torch.squeeze(self())
            loss = self.loss_fn(logits, self.orig_label) + self.l1_loss()
            loss.backward()
            self.optimizer.step()


class AdversarialAttackTrainer:
    def __init__(
        self,
        target_model: StandardAttackableClassifier,
        full_dataset: ud.Dataset,
    ):
        self.target_model = target_model
        self.full_dataset = full_dataset
        self.inferrer = StandardModelInferrer(
            model=target_model, dataset=full_dataset
        )
        # self.l1_beta = l1_beta
        # self.optimizer = optimizer

    # If we want to modify model target_model params, can't use cached_property
    @cached_property
    def correct_predictions_dataset(self) -> ud.Dataset:
        inference_results = self.inferrer.infer()
        return ud.Subset(
            dataset=self.full_dataset,
            indices=inference_results.correct_prediction_indices[0],
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
    start = datetime.now()
    attack_trainer = AdversarialAttackTrainer(
        target_model=model, full_dataset=dataset
    )

    attacker = AdversarialAttacker(
        target_model=model,
        target_dataset=ud.Subset(
            dataset=attack_trainer.correct_predictions_dataset, indices=[0]
        ),
        device=cur_device,
    )
    attacker.run_attack(num_epochs=100)
