import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from lstm_model_stc import LSTMSun2018
from lstm_sun_2018_logit_out import LSTMSun2018Logit
from single_sample_feature_perturber import SingleSampleFeaturePerturber

from standard_model_inferrer import StandardModelInferrer
from x19_mort_dataset import X19MortalityDataset


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
        # pretrained_lstm: LSTMSun2018Logit,
        # feature_perturber: SingleSampleFeaturePerturber,
    ):
        super(AdversarialAttacker, self).__init__()
        self._device = device
        # TODO check if pretrained_lstm & feature_perturber really need to be
        #  instantiated here or if can be instantiated by constructor caller
        #  and passed as params
        self.pretrained_lstm = LSTMSun2018Logit(model_device=cur_device)
        my_checkpoint_path = Path(
            "/home/duane/dproj/UIUC-DLH/project/ehr_adversarial_attack/data"
            "/training_results/2023-04-30_18:49:09.556432.tar"
        )
        my_checkpoint = torch.load(my_checkpoint_path)
        self.pretrained_lstm.load_state_dict(
            my_checkpoint["model_state_dict"], strict=False
        )
        # TODO find way to avoid hard-coding feature_dims
        #  (get sample feature shape)
        self.feature_perturber = SingleSampleFeaturePerturber(
            device=device, feature_dims=(1, 19, 48)
        )

        # for param in self._pretrained_lstm.parameters():
        #     param.requires_grad = False
        # self.pretrained_lstm.eval()
        #
        # self.feature_perturber.train()

        self.to(self._device)

    def forward(self, feature: torch.tensor) -> torch.tensor:
        perturbed_feature = self.feature_perturber(feature)
        logits = self.pretrained_lstm(perturbed_feature)
        return perturbed_feature, logits


class AdversarialAttackTrainer:
    def __init__(
        self,
        device: torch.device,
        attacker: AdversarialAttacker,
        dataset: Dataset,
        kappa: float = 0,
        l1_beta: float = 1,
    ):
        self._device = device
        self._attacker = attacker
        self._optimizer = torch.optim.Adam(
            params=attacker.parameters(), lr=0.001
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

    def train_attacker(self, epochs_per_sample: int):
        dataloader = self._build_single_sample_data_loader()
        # self._attacker.feature_perturber.train()
        self._attacker.train()
        for param in self._attacker.pretrained_lstm.parameters():
            param.requires_grad = False
        # self._attacker.pretrained_lstm.eval()

        all_best_perturbation = torch.FloatTensor()
        all_orig_feature = torch.FloatTensor()
        all_num_adv_examples_found = torch.LongTensor()

        for num_batches, (orig_features, orig_label) in enumerate(dataloader):
            self._attacker.feature_perturber.reset_parameters(
                orig_feature=orig_features
            )
            lowest_loss = torch.inf
            best_perturbation = torch.zeros_like(orig_features)
            orig_features, correct_label = orig_features.to(
                self._device
            ), orig_label.to(self._device)
            num_adv_examples_found = 0
            perturbed_features = torch.clone(orig_features)
            for epoch in range(epochs_per_sample):
                self._optimizer.zero_grad()
                perturbed_features, logits = self._attacker(perturbed_features)
                loss = (
                    self._loss_fn(
                        logits=logits,
                        orig_label=orig_label.item(),
                        kappa=self._kappa,
                    )
                    + self._l1_loss()
                )
                if (loss.item() < lowest_loss) and (
                    (logits[0][int(not orig_label)] - self._kappa)
                    > logits[0][orig_label]
                ):
                    lowest_loss = loss.item()
                    best_perturbation = self._attacker.feature_perturber.perturbation.detach().to(
                        "cpu"
                    )
                    num_adv_examples_found += 1
                loss.backward()
                self._optimizer.step()
            all_best_perturbation = torch.cat(
                (all_best_perturbation, best_perturbation)
            )
            all_orig_feature = torch.cat(
                (all_orig_feature, orig_features.to("cpu"))
            )
            all_num_adv_examples_found = torch.cat(
                (
                    all_num_adv_examples_found,
                    torch.tensor([num_adv_examples_found]),
                )
            )
        return (
            all_num_adv_examples_found,
            all_best_perturbation,
            all_orig_feature,
        )


if __name__ == "__main__":
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    predictive_model = LSTMSun2018(model_device=cur_device)
    checkpoint_path = Path(
        "/home/duane/dproj/UIUC-DLH/project/ehr_adversarial_attack/data"
        "/training_results/2023-04-30_18:49:09.556432.tar"
    )
    checkpoint = torch.load(checkpoint_path)
    predictive_model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )

    full_dataset = X19MortalityDataset()

    # small_dataset = Subset(dataset=full_dataset, indices=[2, 3])

    inferrer = StandardModelInferrer(
        model=predictive_model, dataset=full_dataset
    )

    correctly_predicted_data = inferrer.get_correctly_predicted_samples()

    small_correctly_predicted_data = Subset(
        dataset=correctly_predicted_data, indices=list(range(10))
    )

    x19_lstm_attacker = AdversarialAttacker(device=cur_device)

    trainer = AdversarialAttackTrainer(
        device=cur_device,
        attacker=x19_lstm_attacker,
        dataset=small_correctly_predicted_data,
    )

    result = trainer.train_attacker(epochs_per_sample=100)
