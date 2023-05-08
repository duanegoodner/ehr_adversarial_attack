import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import DataLoader
from dataset_with_index import DatasetWithIndex
from lstm_sun_2018_logit_out import LSTMSun2018Logit
from single_sample_feature_perturber import SingleSampleFeaturePerturber


class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()

    # TODO consider making orig_label a 1-element tensor
    def forward(
        self, logits: torch.tensor, orig_label: int, kappa: torch.tensor
    ) -> torch.tensor:
        # TODO need to modify indexing if use batch_size > 1
        return torch.max(
            logits[0][orig_label] - logits[0][int(not orig_label)], -1 * kappa
        )


class AdversarialAttacker(nn.Module):
    def __init__(
        self,
        device: torch.device,
        feature_perturber: SingleSampleFeaturePerturber,
        logitout_model: LSTMSun2018Logit,
    ):
        super(AdversarialAttacker, self).__init__()
        self._device = device
        self.feature_perturber = feature_perturber
        self.logitout_model = logitout_model
        self.to(self._device)

    def forward(self, feature: torch.tensor) -> torch.tensor:
        perturbed_feature = self.feature_perturber(feature)
        logits = self.logitout_model(perturbed_feature)
        return perturbed_feature, logits


@dataclass
class AdversarialExamplesSummary:
    dataset: DatasetWithIndex | Path
    indices: torch.tensor = None
    num_nonzero_perturbation_elements: torch.tensor = None
    loss_vals: torch.tensor = None
    perturbations: torch.tensor = None

    def __post_init__(self):
        if self.indices is None:
            self.indices = torch.LongTensor()
        if self.num_nonzero_perturbation_elements is None:
            self.num_nonzero_perturbation_elements = torch.LongTensor()
        if self.loss_vals is None:
            self.loss_vals = torch.FloatTensor()
        if self.perturbations is None:
            self.perturbations = torch.FloatTensor()

    def update(
        self,
        index: torch.tensor,
        loss: torch.tensor,
        perturbation: torch.tensor,
    ):
        self.indices = torch.cat(
            (self.indices, index.detach().to("cpu")), dim=0
        )
        self.num_nonzero_perturbation_elements = torch.cat(
            (
                self.num_nonzero_perturbation_elements,
                torch.count_nonzero(perturbation)[None],
            ),
            dim=0,
        )
        self.loss_vals = torch.cat((self.loss_vals, loss.detach().to("cpu")))
        self.perturbations = torch.cat(
            (self.perturbations, perturbation.detach().detach().to("cpu")),
            dim=0,
        )

    @property
    def samples_with_adv_example(self) -> np.ndarray:
        return np.unique(self.indices)

    @property
    def num_samples_with_adv_example(self) -> int:
        return self.samples_with_adv_example.shape[0]


class AdversarialAttackTrainer:
    def __init__(
        self,
        device: torch.device,
        attacker: AdversarialAttacker,
        dataset: DatasetWithIndex,
        learning_rate: float,
        kappa: float,
        l1_beta: float,
        epochs_per_batch: int,
        adv_examples_summary: AdversarialExamplesSummary = None,
    ):
        self._device = device
        self._attacker = attacker
        self._epochs_per_batch = epochs_per_batch
        if adv_examples_summary is None:
            adv_examples_summary = AdversarialExamplesSummary(dataset=dataset)
        self.adv_examples_summary = adv_examples_summary
        self._optimizer = torch.optim.Adam(
            params=attacker.parameters(), lr=learning_rate
        )
        self._loss_fn = AdversarialLoss()
        self._dataset = dataset
        self._kappa = torch.tensor([kappa], dtype=torch.float32).to(
            self._device
        )
        self._l1_beta = l1_beta

    def _build_single_sample_data_loader(self) -> DataLoader:
        return DataLoader(dataset=self._dataset, batch_size=1, shuffle=False)

    def _l1_loss(self):
        return self._l1_beta * torch.norm(
            self._attacker.feature_perturber.perturbation
        )

    #     attacker backprop does not work if call .eval() on logitout,
    #     so need to be sure logitout does not have any dropout layers
    def _set_attacker_to_train_mode(self):
        self._attacker.train()
        for param in self._attacker.logitout_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _apply_bounded_soft_threshold(self, orig_features: torch.tensor):
        perturbation_min = -1 * orig_features
        perturbation_max = torch.ones_like(orig_features) - orig_features

        zero_mask = (
            torch.abs(self._attacker.feature_perturber.perturbation)
            <= self._l1_beta
        )
        self._attacker.feature_perturber.perturbation[zero_mask] = 0

        pos_mask = (
            self._attacker.feature_perturber.perturbation > self._l1_beta
        )
        self._attacker.feature_perturber.perturbation[
            pos_mask
        ] -= self._l1_beta

        neg_mask = (
            self._attacker.feature_perturber.perturbation < -1 * self._l1_beta
        )
        self._attacker.feature_perturber.perturbation[
            neg_mask
        ] += self._l1_beta

        clamped_perturbation = torch.clamp(
            input=self._attacker.feature_perturber.perturbation.data,
            min=perturbation_min,
            max=perturbation_max,
        )

        self._attacker.feature_perturber.perturbation.data.copy_(
            clamped_perturbation
        )

    # Currently require batch size == 1
    def _attack_batch(
        self,
        idx: torch.tensor,
        orig_features: torch.tensor,
        orig_label: torch.tensor,
    ):
        self._attacker.feature_perturber.reset_parameters()
        orig_features, correct_label = orig_features.to(
            self._device
        ), orig_label.to(self._device)
        for epoch in range(self._epochs_per_batch):
            self._optimizer.zero_grad()
            perturbed_features, logits = self._attacker(orig_features)
            loss = (
                self._loss_fn(
                    logits=logits,
                    orig_label=orig_label.item(),
                    kappa=self._kappa,
                )
                + self._l1_loss()
            )
            if (logits[0][int(not orig_label)] - self._kappa) > logits[0][
                orig_label
            ]:
                self.adv_examples_summary.update(
                    index=idx,
                    loss=loss,
                    perturbation=self._attacker.feature_perturber.perturbation.detach().to(
                        "cpu"
                    ),
                )
            loss.backward()
            self._optimizer.step()
            self._apply_bounded_soft_threshold(orig_features=orig_features)

    def train_attacker(self):
        dataloader = self._build_single_sample_data_loader()
        self._set_attacker_to_train_mode()
        for num_batches, (idx, orig_features, orig_label) in enumerate(
            dataloader
        ):
            self._attack_batch(
                idx=idx, orig_features=orig_features, orig_label=orig_label
            )
