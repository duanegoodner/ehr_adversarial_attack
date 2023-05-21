import numpy as np
import optuna
import torch
import torch.nn as nn
from dataclasses import dataclass
from optuna.trial import TrialState
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Any, Callable, TypedDict
from lstm_model_stc import BidirectionalX19LSTM
from standard_model_trainer import StandardModelTrainer
from weighted_dataloader_builder import (
    DataLoaderBuilder,
    WeightedDataLoaderBuilder,
)
import os
import torch.nn.functional as F
import torch.optim as optim


@dataclass
class CVDataSets:
    train: list[Dataset]
    validation: list[Dataset]


@dataclass
class TrainEvalDatasetPair:
    train: Dataset
    validation: Dataset


@dataclass
class X19LSTMHyperParameterSettings:
    log_lstm_hidden_size: int
    lstm_act_name: str
    dropout: float
    log_fc_hidden_size: int
    fc_act_name: str
    optimizer_name: str
    learning_rate: float
    log_batch_size: int


@dataclass
class X19MLSTMTuningRanges:
    log_lstm_hidden_size: tuple[int, int]
    lstm_act_options: tuple[str, ...]
    dropout: tuple[float, float]
    log_fc_hidden_size: tuple[int, int]
    fc_act_options: tuple[str, ...]
    optimizer_options: tuple[str, ...]
    learning_rate: tuple[float, float]
    log_batch_size: tuple[int, int]

    def get_param_vals(self, trial) -> X19LSTMHyperParameterSettings:
        return X19LSTMHyperParameterSettings(
            log_lstm_hidden_size=trial.suggest_int(
                "log_lstm_hidden_size", *self.log_lstm_hidden_size
            ),
            lstm_act_name=trial.suggest_categorical(
                "lstm_act", list(self.lstm_act_options)
            ),
            dropout=trial.suggest_float("dropout", *self.dropout),
            log_fc_hidden_size=trial.suggest_int(
                "log_fc_hidden_size", *self.log_lstm_hidden_size
            ),
            fc_act_name=trial.suggest_categorical(
                "fc_act", list(self.fc_act_options)
            ),
            optimizer_name=trial.suggest_categorical(
                "optimizer", list(self.optimizer_options)
            ),
            learning_rate=trial.suggest_float(
                "lr", *self.learning_rate, log=True
            ),
            log_batch_size=trial.suggest_int(
                "log_batch_size", *self.log_batch_size
            ),
        )


class HyperParameterTuner:
    def __init__(
        self,
        device: torch.device,
        dataset: Dataset,
        collate_fn: Callable,
        num_folds: int,
        num_cv_epochs: int,
        epochs_per_fold: int,
        tuning_ranges: X19MLSTMTuningRanges,
        fold_class: Callable = StratifiedKFold,
        train_loader_builder=WeightedDataLoaderBuilder(),
        fold_generator_builder_random_seed: int = 1234,
        weighted_dataloader_random_seed: int = 22,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        performance_metric: str = "roc_auc",
    ):
        self.device = device
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.num_folds = num_folds
        self.num_cv_epochs = num_cv_epochs
        self.epochs_per_fold = epochs_per_fold
        self.fold_class = fold_class
        self.fold_generator_random_seed = fold_generator_builder_random_seed
        self.fold_generator_builder = fold_class(
            n_splits=num_folds,
            shuffle=True,
            random_state=fold_generator_builder_random_seed,
        )
        self.cv_datasets = self.create_datasets()
        self.train_loader_builder = train_loader_builder
        self.loss_fn = loss_fn
        self.performance_metric = performance_metric
        self.tuning_ranges = tuning_ranges
        self.weighted_dataloader_random_seed = weighted_dataloader_random_seed

    def create_datasets(self) -> list[TrainEvalDatasetPair]:
        fold_generator = self.fold_generator_builder.split(
            self.dataset[:][0], self.dataset[:][1]
        )

        all_train_eval_pairs = []

        for fold_idx, (train_indices, validation_indices) in enumerate(
            fold_generator
        ):
            train_dataset = Subset(dataset=self.dataset, indices=train_indices)
            validation_dataset = Subset(
                dataset=self.dataset, indices=validation_indices
            )
            all_train_eval_pairs.append(
                TrainEvalDatasetPair(
                    train=train_dataset, validation=validation_dataset
                )
            )

        return all_train_eval_pairs

    @staticmethod
    def define_model(settings: X19LSTMHyperParameterSettings):
        return nn.Sequential(
            BidirectionalX19LSTM(
                input_size=19,
                lstm_hidden_size=2**settings.log_lstm_hidden_size,
            ),
            getattr(nn, settings.lstm_act_name)(),
            nn.Dropout(p=settings.dropout),
            nn.Linear(
                in_features=2 * (2**settings.log_lstm_hidden_size),
                out_features=2**settings.log_fc_hidden_size,
            ),
            getattr(nn, settings.fc_act_name)(),
            nn.Linear(
                in_features=2**settings.log_fc_hidden_size, out_features=2
            ),
            nn.Softmax(dim=1),
        )

    def create_trainers(self, trial):
        settings = self.tuning_ranges.get_param_vals(trial)

        trainers = []
        for dataset_pair in self.cv_datasets:
            model = self.define_model(settings=settings)
            train_loader = self.train_loader_builder.build(
                dataset=dataset_pair.train,
                batch_size=2**settings.log_batch_size,
                collate_fn=self.collate_fn,
            )
            validation_loader = DataLoader(
                dataset=dataset_pair.validation,
                batch_size=128,
                shuffle=False,
                collate_fn=self.collate_fn,
            )

            trainer = StandardModelTrainer(
                train_device=self.device,
                eval_device=self.device,
                model=model,
                loss_fn=self.loss_fn,
                optimizer=getattr(torch.optim, settings.optimizer_name)(
                    model.parameters(), lr=settings.learning_rate
                ),
                train_loader=train_loader,
                test_loader=validation_loader,
            )

            trainers.append(trainer)

        return trainers

    def objective_fn(self, trial):
        trainers = self.create_trainers(trial)
        all_folds_metric_of_interest = torch.zeros(
            self.num_folds, dtype=torch.float32
        )
        for cv_epoch_idx in range(self.num_cv_epochs):
            for trainer_idx, trainer in enumerate(trainers):
                trainer.train_model(num_epochs=self.epochs_per_fold)
                metrics = trainer.evaluate_model()
                metric_of_interest = getattr(metrics, self.performance_metric)
                all_folds_metric_of_interest[trainer_idx] += metric_of_interest
                trainer.model.to("cpu")

        return (
            torch.mean(all_folds_metric_of_interest).item()
            / self.num_cv_epochs
        )

    def tune(self, n_trials: int = 10, timeout: int | None = None):
        study = optuna.create_study(direction="maximize")
        study.optimize(
            func=self.objective_fn, n_trials=n_trials, timeout=timeout
        )

        pruned_trials = study.get_trials(
            deepcopy=False, states=[TrialState.PRUNED]
        )
        complete_trials = study.get_trials(
            deepcopy=False, states=[TrialState.COMPLETE]
        )

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
