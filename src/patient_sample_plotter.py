import numpy as np
import torch.cuda
from torch.utils.data import DataLoader, Dataset
from cv_trainer import CrossValidationTrainer
from x19_mort_dataset import X19MortalityDataset
from lstm_model import BinaryBidirectionalLSTM


class PatientSamplePlotter:
    def __init__(self, x19_mort_dataloader: DataLoader):
        self._data_loader = x19_mort_dataloader


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
