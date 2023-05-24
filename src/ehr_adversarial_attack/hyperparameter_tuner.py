import numpy as np
import optuna
import torch
import torch.nn as nn
from dataclasses import dataclass
from datetime import datetime
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Callable
import project_config as pc
import resource_io as rio
from data_structures import CVTrialSummary, TrainEvalLogs
from lstm_model_stc import BidirectionalX19LSTM
from standard_model_trainer import StandardModelTrainer
from torch.utils.tensorboard import SummaryWriter
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
class X19MLSTMTuningRanges:
    log_lstm_hidden_size: tuple[int, int]
    lstm_act_options: tuple[str, ...]
    dropout: tuple[float, float]
    log_fc_hidden_size: tuple[int, int]
    fc_act_options: tuple[str, ...]
    optimizer_options: tuple[str, ...]
    learning_rate: tuple[float, float]
    log_batch_size: tuple[int, int]


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

    @classmethod
    def from_optuna(cls, trial, tuning_ranges: X19MLSTMTuningRanges):
        return cls(
            log_lstm_hidden_size=trial.suggest_int(
                "log_lstm_hidden_size",
                *tuning_ranges.log_lstm_hidden_size,
            ),
            lstm_act_name=trial.suggest_categorical(
                "lstm_act", list(tuning_ranges.lstm_act_options)
            ),
            dropout=trial.suggest_float("dropout", *tuning_ranges.dropout),
            log_fc_hidden_size=trial.suggest_int(
                "log_fc_hidden_size", *tuning_ranges.log_fc_hidden_size
            ),
            fc_act_name=trial.suggest_categorical(
                "fc_act", list(tuning_ranges.fc_act_options)
            ),
            optimizer_name=trial.suggest_categorical(
                "optimizer", list(tuning_ranges.optimizer_options)
            ),
            learning_rate=trial.suggest_float(
                "lr", *tuning_ranges.learning_rate, log=True
            ),
            log_batch_size=trial.suggest_int(
                "log_batch_size", *tuning_ranges.log_batch_size
            ),
        )


class HyperParameterTuner:
    def __init__(
        self,
        device: torch.device,
        dataset: Dataset,
        collate_fn: Callable,
        num_trials: int,
        num_folds: int,
        num_cv_epochs: int,
        epochs_per_fold: int,
        tuning_ranges: dataclass,
        fold_class: Callable = StratifiedKFold,
        train_loader_builder=WeightedDataLoaderBuilder(),
        fold_generator_builder_random_seed: int = 1234,
        weighted_dataloader_random_seed: int = 22,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        performance_metric: str = "roc_auc",
        sampler: optuna.samplers.BaseSampler = TPESampler(),
        pruner: optuna.pruners.BasePruner = MedianPruner(),
        output_dir: Path = None,
        save_trial_info: bool = False
    ):
        self.device = device
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.num_trials = num_trials
        self.num_folds = num_folds
        self.num_cv_epochs = num_cv_epochs
        self.epochs_per_fold = epochs_per_fold
        self.fold_class = fold_class
        self.fold_generator_builder_random_seed = (
            fold_generator_builder_random_seed
        )
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
        self.sampler = sampler
        self.pruner = pruner
        self.output_dir = self.initialize_output_dir(output_dir=output_dir)
        self.exporter = rio.ResourceExporter()
        self.save_trial_info = save_trial_info

    @staticmethod
    def initialize_output_dir(output_dir: Path = None) -> Path:
        if output_dir is None:
            dirname = f"{datetime.now()}".replace(" ", "_")
            output_dir = pc.HYPERPARAMETER_TUNING_OUT_DIR / dirname
        assert not output_dir.exists()
        output_dir.mkdir()
        return output_dir

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

    @staticmethod
    def initialize_model(model: nn.Module):
        for name, param in model.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def create_trainer(
        self,
        train_eval_pair: TrainEvalDatasetPair,
        settings: X19LSTMHyperParameterSettings,
        model: nn.Module,
        summary_writer: SummaryWriter,
        fold_num: int,
    ):
        train_loader = self.train_loader_builder.build(
            dataset=train_eval_pair.train,
            batch_size=2**settings.log_batch_size,
            collate_fn=self.collate_fn,
        )
        validation_loader = DataLoader(
            dataset=train_eval_pair.validation,
            batch_size=128,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

        return StandardModelTrainer(
            train_device=self.device,
            eval_device=self.device,
            model=model,
            loss_fn=self.loss_fn,
            optimizer=getattr(torch.optim, settings.optimizer_name)(
                model.parameters(), lr=settings.learning_rate
            ),
            train_loader=train_loader,
            test_loader=validation_loader,
            summary_writer=summary_writer,
            summary_writer_label=f"fold_{fold_num}",
        )

    def create_trainers(
        self,
        settings: X19LSTMHyperParameterSettings,
        summary_writer: SummaryWriter,
        trial_number: int,
    ):
        trainers = []
        for fold_idx, dataset_pair in enumerate(self.cv_datasets):
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
                summary_writer=summary_writer,
                summary_writer_group=str(trial_number),
                summary_writer_subgroup=str(fold_idx),
                checkpoint_dir=self.output_dir / "checkpoints",
            )

            trainers.append(trainer)

        return trainers

    def export_trial_info(self, trial, trainers: list[StandardModelTrainer]):
        trial_summary = CVTrialSummary(
            trial=trial,
            logs=[
                TrainEvalLogs(train=trainer.train_log, eval=trainer.eval_log)
                for trainer in trainers
            ],
        )

        self.exporter.export(
            resource=trial_summary,
            path=self.output_dir / f"trial_{trial.number}_summary.pickle",
        )

    def objective_fn(self, trial) -> float | None:
        settings = X19LSTMHyperParameterSettings.from_optuna(
            trial=trial, tuning_ranges=self.tuning_ranges
        )

        summary_writer_path = (
            self.output_dir / "tensorboard" / f"Trial_{trial.number}"
        )
        summary_writer = SummaryWriter(str(summary_writer_path))

        trainers = self.create_trainers(
            settings=settings,
            summary_writer=summary_writer,
            trial_number=trial.number,
        )

        all_epoch_eval_results = []
        all_epoch_metric_of_interest = []

        # TODO: think more abot this seed. Is it needed???
        # Do we want each trial to have same sample selection pattern?
        # torch.manual_seed(self.weighted_dataloader_random_seed)

        for cv_epoch in range(self.num_cv_epochs):
            epoch_results = []
            all_folds_metric_of_interest = []
            for fold_idx, trainer in enumerate(trainers):
                trainer.train_model(num_epochs=self.epochs_per_fold)
                eval_results = trainer.evaluate_model()
                epoch_results.append(eval_results)
                all_folds_metric_of_interest.append(
                    getattr(eval_results, self.performance_metric)
                )
                trainer.model.to("cpu")
            all_epoch_eval_results.append(epoch_results)
            all_epoch_metric_of_interest.append(
                np.mean(all_folds_metric_of_interest)
            )

        if self.save_trial_info:
            self.export_trial_info(trial=trial, trainers=trainers)

        return min(all_epoch_metric_of_interest)

    def tune(self, timeout: int | None = None) -> optuna.Study:
        study = optuna.create_study(
            direction="maximize", sampler=self.sampler, pruner=self.pruner
        )
        study.optimize(
            func=self.objective_fn, n_trials=self.num_trials, timeout=timeout
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

        timestamp = str(datetime.now()).replace(" ", "_")
        study_filename = f"optuna_study_{timestamp}.pickle"
        study_export_path = self.output_dir / study_filename
        self.exporter.export(resource=study, path=study_export_path)
        hyperparameter_tuner_filename = (
            f"hyperparameter_tuner_{timestamp}.pickle"
        )
        hyperparameter_tuner_path = (
            self.output_dir / hyperparameter_tuner_filename
        )
        self.exporter.export(resource=self, path=hyperparameter_tuner_path)

        return study
