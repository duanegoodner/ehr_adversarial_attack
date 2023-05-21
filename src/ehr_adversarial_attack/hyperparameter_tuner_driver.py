import torch
from x19_mort_general_dataset import x19m_collate_fn, X19MGeneralDataset
from hyperparameter_tuner import HyperParameterTuner


def main():
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    tuner = HyperParameterTuner(
        device=cur_device,
        dataset=X19MGeneralDataset.from_feaure_finalizer_output(),
        collate_fn=x19m_collate_fn,
        num_folds=2,
        epochs_per_fold=2
    )

    tuner.tune(n_trials=5, timeout=600)


if __name__ == "__main__":
    main()






