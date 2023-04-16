from torch.utils.data import Subset, DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import KFold
import numpy as np
import torch
import torch.nn as nn


class CrossValidationTrainer:
    def __init__(
        self,
        dataset: Dataset,
        model: nn.Module,  # consider more specific interface class
        num_folds: int,
        batch_size: int,
        epochs_per_fold: int,
        global_epochs: int,
    ):
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

    @staticmethod
    def build_train_sampler(y_train: list[torch.tensor]):
        class_sample_counts = np.unique(y_train, return_counts=True)[1]
        weight = 1.0 / class_sample_counts
        sample_weights = torch.from_numpy(
            np.array([weight[t] for t in y_train])
        )
        train_sampler = WeightedRandomSampler(
            sample_weights.type("torch.DoubleTensor"), len(sample_weights)
        )
        return train_sampler

    def train_fold(self, train_indices: np.ndarray):
        train_split = Subset(dataset=self.dataset, indices=train_indices)
        y_train = [self.dataset[i][1] for i in train_split.indices]
        train_sampler = self.build_train_sampler(y_train=y_train)
        train_dataloader = DataLoader(
            dataset=train_split,
            batch_size=self.batch_size,
            sampler=train_sampler,
        )
        # print("train model with this split")
        # for batch_idx, (data, target) in enumerate(train_dataloader):
        #     print(batch_idx)
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
            for fold_epoch in range(self.epochs_per_fold):
                self.train_fold(train_indices=train_indices)
            self.evaluate_fold(validation_indices=validation_indices)

    def run(self):
        for global_epoch in range(self.global_epochs):
            self.run_one_global_epoch()


# if __name__ == "__main__":
#     cv_trainer = CrossValidationTrainer(
#         dataset=X19MortalityDataset(),
#         model=BinaryBidirectionalLSTM(
#             input_size=48, lstm_hidden_size=128, fc_hidden_size=32
#         ),
#         num_folds=5,
#         batch_size=128,
#         epochs_per_fold=2,
#         global_epochs=1)
#     cv_trainer.run()
