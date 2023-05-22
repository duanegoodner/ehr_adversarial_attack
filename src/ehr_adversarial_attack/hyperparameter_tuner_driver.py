import time
import torch
from x19_mort_general_dataset import x19m_collate_fn, X19MGeneralDataset
from hyperparameter_tuner import (
    HyperParameterTuner,
    optuna_report,
    ray_tune_report,
    X19MLSTMTuningRanges,
    X19LSTMHyperParameterSettings,
)


my_tuning_ranges = X19MLSTMTuningRanges(
    log_lstm_hidden_size=(5, 7),
    lstm_act_options=("ReLU", "Tanh"),
    dropout=(0, 0.5),
    log_fc_hidden_size=(4, 8),
    fc_act_options=("ReLU", "Tanh"),
    optimizer_options=("Adam", "RMSprop", "SGD"),
    learning_rate=(1e-5, 1e-1),
    log_batch_size=(5, 8),
)


def main():
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    tuner = HyperParameterTuner(
        device=cur_device,
        dataset=X19MGeneralDataset.from_feaure_finalizer_output(),
        collate_fn=x19m_collate_fn,
        num_folds=5,
        num_cv_epochs=2,
        epochs_per_fold=5,
        tuning_ranges=my_tuning_ranges,
        setting_retriever=X19LSTMHyperParameterSettings.from_optuna,
        metric_reporter=optuna_report,
        return_metric_from_objective=True
    )

    tuner.optuna_tune(n_trials=20, timeout=None)


if __name__ == "__main__":
    ray_tune_config = my_tuning_ranges.to_ray_tune_config()
    start = time.time()
    main()
    end = time.time()

    print(f"total time = {end - start}")
