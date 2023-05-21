import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Callable
from lstm_model_stc import BidirectionalX19LSTM
from standard_model_trainer import StandardModelTrainer
from weighted_dataloader_builder import (
    DataLoaderBuilder,
    WeightedDataLoaderBuilder,
)


import os

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class HyperParameterTuner:
    def __init__(
        self,
        device: torch.device,
        dataset: Dataset,
        collate_fn: Callable,
        num_folds: int,
        epochs_per_fold: int,
        fold_class: Callable = StratifiedKFold,
        train_loader_builder=WeightedDataLoaderBuilder(),
        log_lstm_hidden_size_range: tuple[int, int] = (5, 7),
        lstm_act_options: tuple = ("ReLU", "Tanh"),
        dropout_range: tuple[float, float] = (0, 0.5),
        log_fc_hidden_size_range: tuple[int, int] = (4, 8),
        fc_act_options: tuple = ("ReLU", "Tanh"),
        optimizer_options: tuple = ("Adam", "RMSprop", "SGD"),
        learning_rate_range: tuple[float, float] = (1e-5, 1e-1),
        log_batch_size_range: tuple[int, int] = (5, 8),
        fold_generator_builder_random_seed: int = 1234,
        weighted_dataloader_random_seed: int = 22,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        performance_metric: str = "roc_auc",
    ):
        self.device = device
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.num_folds = num_folds
        self.epochs_per_fold = epochs_per_fold
        self.fold_class = fold_class
        self.fold_generator_random_seed = fold_generator_builder_random_seed
        self.fold_generator_builder = fold_class(
            n_splits=num_folds,
            shuffle=True,
            random_state=fold_generator_builder_random_seed,
        )
        self.fold_generator = self.fold_generator_builder.split(
            dataset[:][0], dataset[:][1]
        )
        self.train_loader_builder = train_loader_builder
        self.log_lstm_hidden_size_range = log_lstm_hidden_size_range
        self.lstm_act_options = lstm_act_options
        self.dropout_range = dropout_range
        self.log_fc_hidden_size_range = log_fc_hidden_size_range
        self.fc_act_options = fc_act_options
        self.learning_rate_range = learning_rate_range
        self.log_batch_size_range = log_batch_size_range
        self.optimizer_options = optimizer_options
        self.weighted_dataloader_random_seed = weighted_dataloader_random_seed
        self.loss_fn = loss_fn
        self.performance_metric = performance_metric

    def define_model(self, trial):
        log_lstm_hidden_size = trial.suggest_int(
            "log_lstm_hidden_size", *self.log_lstm_hidden_size_range
        )
        lstm_act_name = trial.suggest_categorical(
            "lstm_act", list(self.lstm_act_options)
        )
        log_fc_hidden_size = trial.suggest_int(
            "log_fc_hidden_size", *self.log_lstm_hidden_size_range
        )
        fc_act_name = trial.suggest_categorical(
            "fc_act", list(self.fc_act_options)
        )

        dropout = trial.suggest_float(
            "dropout", *self.dropout_range, log=False
        )

        return nn.Sequential(
            BidirectionalX19LSTM(
                input_size=19, lstm_hidden_size=2**log_lstm_hidden_size
            ),
            getattr(nn, lstm_act_name)(),
            nn.Dropout(p=dropout),
            nn.Linear(
                in_features=2 * (2**log_lstm_hidden_size),
                out_features=2**log_fc_hidden_size,
            ),
            getattr(nn, fc_act_name)(),
            nn.Linear(in_features=2**log_fc_hidden_size, out_features=2),
            nn.Softmax(dim=1),
        )

    @staticmethod
    def initialize_model(model: nn.Module):
        for name, param in model.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def objective_fn(self, trial):
        fold_generator = self.fold_generator_builder.split(
            self.dataset[:][0], self.dataset[:][1]
        )
        model = self.define_model(trial)
        lr = trial.suggest_float("lr", *self.learning_rate_range, log=True)
        optimizer_name = trial.suggest_categorical(
            "optimizer", list(self.optimizer_options)
        )
        optimizer = getattr(torch.optim, optimizer_name)(
            model.parameters(), lr=lr
        )
        log_batch_size = trial.suggest_int(
            "log_batch_size", *self.log_batch_size_range
        )

        all_folds_metric_of_interest = []

        torch.manual_seed(self.weighted_dataloader_random_seed)
        for fold_idx, (train_indices, test_indices) in enumerate(
            fold_generator
        ):
            self.initialize_model(model=model)
            train_dataset = Subset(dataset=self.dataset, indices=train_indices)
            train_loader = self.train_loader_builder.build(
                dataset=train_dataset,
                batch_size=2**log_batch_size,
                collate_fn=self.collate_fn,
            )
            validation_dataset = Subset(
                dataset=self.dataset, indices=test_indices
            )
            validation_loader = DataLoader(
                dataset=validation_dataset,
                batch_size=128,
                shuffle=False,
                collate_fn=self.collate_fn,
            )
            trainer = StandardModelTrainer(
                device=self.device,
                model=model,
                loss_fn=self.loss_fn,
                optimizer=optimizer,
                train_loader=train_loader,
                test_loader=validation_loader,
            )
            trainer.train_model(num_epochs=self.epochs_per_fold)
            metrics = trainer.evaluate_model()
            metric_of_interest = getattr(metrics, self.performance_metric)
            all_folds_metric_of_interest.append(metric_of_interest)
        return np.mean(all_folds_metric_of_interest)

    def tune(self, n_trials: int = 10, timeout: int = 600):
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
