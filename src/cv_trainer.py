from torch.utils.data import Subset, DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import KFold
import numpy as np
import torch
import torch.nn as nn
from x19_mort_dataset import X19MortalityDataset
from lstm_model import BinaryBidirectionalLSTM


class WeightedRandomSamplerBuilder:
    def __init__(self, skewed_features: torch.tensor):
        assert skewed_features.dim() == 1
        assert not torch.is_floating_point(skewed_features)
        assert not torch.is_complex(skewed_features)
        self._features = skewed_features

    def build(self):
        class_sample_counts = np.unique(self._features, return_counts=True)[1]
        class_weights = 1.0 / class_sample_counts
        sample_weights = np.choose(self._features, class_weights)
        return WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights)
        )


class CrossValidationTrainer:
    def __init__(
        self,
        device: torch.device,
        dataset: Dataset,
        model: nn.Module,  # consider more specific interface class
        num_folds: int,
        batch_size: int,
        epochs_per_fold: int,
        global_epochs: int,
    ):
        self.device = device
        self.dataset = dataset
        self.model = model
        self.num_folds = num_folds
        self.batch_size = batch_size
        self.fold_generator = KFold(n_splits=num_folds, shuffle=True)
        self.epochs_per_fold = epochs_per_fold
        self.global_epochs = global_epochs

    @property
    def dataset_size(self) -> int:
        return len(self.dataset)

    def train_fold(self, train_indices: np.ndarray):
        train_split = Subset(dataset=self.dataset, indices=train_indices)
        train_sampler = WeightedRandomSamplerBuilder(
            skewed_features=self.dataset[train_split.indices][1]
        ).build()
        train_dataloader = DataLoader(
            dataset=train_split,
            batch_size=self.batch_size,
            sampler=train_sampler,
        )
        self.model.train_model(
            train_loader=train_dataloader, num_epochs=self.epochs_per_fold
        )

    def evaluate_fold(self, validation_indices: np.ndarray):
        validation_split = Subset(
            dataset=self.dataset, indices=validation_indices
        )
        validation_dataloader = DataLoader(
            dataset=validation_split, batch_size=self.batch_size, shuffle=True
        )
        # print("evaluate model with this split")
        # for batch_idx, (data, target) in enumerate(validation_dataloader):
        #     print(batch_idx)
        self.model.evaluate_model(test_loader=validation_dataloader)

    def run_one_global_epoch(self):
        for fold_idx, (train_indices, validation_indices) in enumerate(
            self.fold_generator.split(range(self.dataset_size))
        ):
            print(f"    Training fold # {fold_idx}")
            for fold_epoch in range(self.epochs_per_fold):
                print(f"        Running fold_epoch # {fold_epoch}")
                self.train_fold(train_indices=train_indices)
            print(f"    Evaluating fold # {fold_idx}")
            self.evaluate_fold(validation_indices=validation_indices)

    def run(self):
        self.model.to(self.device)
        for global_epoch in range(self.global_epochs):
            print(f"Starting global epoch # {global_epoch}")
            print("-------------------------\n")
            self.run_one_global_epoch()


if __name__ == "__main__":
    if torch.cuda.is_available():
        my_device = torch.device("cuda:0")
    else:
        my_device = torch.device("cpu")
    cv_trainer = CrossValidationTrainer(
        device=my_device,
        dataset=X19MortalityDataset(),
        model=BinaryBidirectionalLSTM(
            device=my_device,
            input_size=48,
            lstm_hidden_size=128,
            fc_hidden_size=32,
        ),
        num_folds=5,
        batch_size=128,
        epochs_per_fold=2,
        global_epochs=1,
    )
    cv_trainer.run()
