import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader, Dataset
import standard_model_trainer as smt
from weighted_dataloader_builder import (
    DataLoaderBuilder,
    WeightedDataLoaderBuilder,
)


class CrossValidator:
    def __init__(
        self,
        dataset: Dataset,
        trainer: smt.StandardModelTrainer,
        num_folds: int,
        batch_size: int,
        epochs_per_fold: int,
        max_global_epochs: int,
        dataloader_builder: DataLoaderBuilder = WeightedDataLoaderBuilder()
    ):
        self.dataset = dataset
        self.trainer = trainer
        self.num_folds = num_folds
        self.fold_generator = KFold(n_splits=num_folds, shuffle=True)
        self.batch_size = batch_size
        self.epochs_per_fold = epochs_per_fold
        self.max_global_epochs = max_global_epochs
        self.dataloader_builder = dataloader_builder

    @property
    def dataset_size(self) -> int:
        return len(self.dataset)

    def train_fold(self, train_indices: np.ndarray):
        train_split = Subset(dataset=self.dataset, indices=train_indices)
        train_dataloader = self.dataloader_builder.build(
            dataset=train_split, batch_size=self.batch_size
        )
        self.trainer.train_model(
            train_loader=train_dataloader,
            num_epochs=self.epochs_per_fold)

    def evaluate_fold(self, validation_indices: np.ndarray):
        validation_split = Subset(
            dataset=self.dataset, indices=validation_indices
        )
        validation_dataloader = DataLoader(
            dataset=validation_split, batch_size=self.batch_size, shuffle=True
        )
        self.trainer.evaluate_model(test_loader=validation_dataloader)

    def run_global_epoch(self):
        for fold_idx, (train_indices, validation_indices) in enumerate(
                self.fold_generator.split(range(self.dataset_size))
        ):
            self.train_fold(train_indices=train_indices)
            self.evaluate_fold(validation_indices=validation_indices)

    def run(self):
        for global_epoch in range(self.max_global_epochs):
            self.run_global_epoch()





