import torch
import torch.nn as nn
from functools import cached_property
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from lstm_sun_2018_logit_out import LSTMSun2018Logit
from standard_model_inferrer import StandardModelInferrer
from x19_mort_dataset import X19MortalityDataset


class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()

    def forward(
        self, logits: torch.tensor, orig_label: int, kappa: torch.tensor
    ) -> torch.tensor:
        return max(logits[orig_label] - logits[int(not orig_label)], kappa)

        # return logits[orig_label] - logits[int(not orig_label)]


class AdversarialAttacker(nn.Module):
    def __init__(
        self,
        target_dataset: Dataset,
        model_device: torch.device,
        kappa: torch.tensor,
    ):
        super(AdversarialAttacker, self).__init__()
        self.model_device = model_device
        self.target_dataset = target_dataset
        self.orig_feature = target_dataset[0][0].to(model_device)
        self.orig_label = target_dataset[0][1].to(model_device)
        self.data_loader = DataLoader(dataset=target_dataset, batch_size=1)
        self.classifier_logit = LSTMSun2018Logit(model_device=model_device)
        self.perturbed_feature = nn.Parameter(
            self.target_dataset[0][0][None, :, :], requires_grad=True
        )
        self.loss_fn = AdversarialLoss()
        self.kappa = kappa.to(model_device)
        self.optimizer = torch.optim.SGD(params=self.parameters(), lr=0.001)
        self.l1_beta = 0.001
        self.to(model_device)

    def forward(self) -> torch.tensor:
        return self.classifier_logit(self.perturbed_feature)

    def l1_loss(self):
        return self.l1_beta * torch.norm(
            self.perturbed_feature - self.orig_feature
        )

    def run_attack(self, num_epochs: int):
        self.train()
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            logits = torch.squeeze(self())
            loss = self.loss_fn(
                logits=logits, orig_label=self.orig_label, kappa=self.kappa
            )  # + self.l1_loss()
            loss.backward()
            self.optimizer.step()


class AdversarialAttackTrainer:
    def __init__(
        self,
        target_model: nn.Module,
        full_dataset: Dataset,
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
    def correct_predictions_dataset(self) -> Dataset:
        inference_results = self.inferrer.infer()
        return Subset(
            dataset=self.full_dataset,
            indices=inference_results.correct_prediction_indices[0],
        )


if __name__ == "__main__":
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    model = LSTMSun2018Logit(model_device=cur_device)
    checkpoint_path = Path(
        "/home/duane/dproj/UIUC-DLH/project/ehr_adversarial_attack/data"
        "/training_results/2023-04-30_18:49:09.556432.tar"
    )
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    dataset = X19MortalityDataset()
    attack_trainer = AdversarialAttackTrainer(
        target_model=model, full_dataset=dataset
    )

    attacker = AdversarialAttacker(
        target_dataset=Subset(
            dataset=attack_trainer.correct_predictions_dataset, indices=[0]
        ),
        model_device=cur_device,
        kappa=torch.tensor(0, requires_grad=True, dtype=torch.float)
    )
    attacker.run_attack(num_epochs=100)
