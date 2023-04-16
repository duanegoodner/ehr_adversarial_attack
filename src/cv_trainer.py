from torch.utils.data import Subset, DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import KFold
import numpy as np
import torch.nn as nn
from lstm_model import BinaryBidirectionalLSTM
from x19_mort_dataset import X19MortalityDataset


class CrossValidationTrainer:
    def __init__(
        self,
        dataset: Dataset,
        model: nn.Module,
        num_folds: int,
        batch_size: int,
        epochs_per_fold: int,
        global_epochs: int,
    ):
        self.dataset = dataset
        self.model = model
        self.num_folds = num_folds
        self.batch_size = batch_size
        self.fold_generator = KFold(
            n_splits=self.num_folds, shuffle=True, random_state=42
        )
        self.epochs_per_fold = epochs_per_fold
        self.global_epochs = global_epochs

    @property
    def full_dataset_labels(self) -> np.ndarray:
        return np.array(
            [
                self.dataset[sample_idx][1]
                for sample_idx in range(len(self.dataset))
            ]
        )

    def calc_label_sampling_weights(
        self, train_indices: np.ndarray
    ) -> np.ndarray:
        train_labels = self.full_dataset_labels[train_indices]
        np_unique_info = np.unique(train_labels, return_counts=True)
        label_sampling_weights = 1 / np_unique_info[1]
        return label_sampling_weights

    def build_train_loader(self, train_indices: np.ndarray) -> DataLoader:
        label_sampling_weights = self.calc_label_sampling_weights(
            train_indices=train_indices
        )
        # train_labels = self.full_dataset_labels[train_indices]
        train_labels = np.take(self.full_dataset_labels, train_indices)
        # train_sampling_weights = np.take(
        #     a=label_sampling_weights, indices=train_labels
        # )

        train_sampling_weights = np.array([0.05, 0.95])
        train_sampler = WeightedRandomSampler(
            weights=train_sampling_weights,
            num_samples=len(train_indices),
            replacement=True,
        )
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
        )

    def train_fold(self, train_indices: np.ndarray):
        train_loader = self.build_train_loader(train_indices)
        for batch_index, (features, labels) in enumerate(train_loader):
            print("do some training")
        # self.model.train_model(
        #     train_loader=train_loader, num_epochs=self.epochs_per_fold
        # )

    def evaluate_fold(self, validation_indices: np.ndarray):
        validation_dataset = Subset(
            dataset=self.dataset, indices=validation_indices
        )
        validation_loader = DataLoader(
            dataset=validation_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        for batch_index, (features, labels) in enumerate(validation_loader):
            print("do some evaluation")
        # self.model.evaluate_model(test_loader=validation_loader)

    def run_cv_training(self):
        for fold_idx, (train_indices, validation_indices) in enumerate(
            self.fold_generator.split(range(len(self.dataset)))
        ):
            self.train_fold(train_indices=train_indices)
            self.evaluate_fold(validation_indices=validation_indices)


if __name__ == "__main__":
    full_dataset = X19MortalityDataset()
    subset_indices = np.random.randint(low=0, high=len(full_dataset), size=1000)
    quick_check_dataset = Subset(
        dataset=full_dataset, indices=subset_indices
    )

    cv_trainer = CrossValidationTrainer(
        dataset=quick_check_dataset,
        model=BinaryBidirectionalLSTM(
            input_size=48, lstm_hidden_size=128, fc_hidden_size=32
        ),
        num_folds=5,
        batch_size=32,
        epochs_per_fold=2,
        global_epochs=1,
    )

    cv_trainer.run_cv_training()
