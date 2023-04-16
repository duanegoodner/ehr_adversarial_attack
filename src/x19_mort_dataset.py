import numpy as np
import pickle
import torch
from pathlib import Path
from torch.utils.data import Dataset
import project_config as pc


def get_pickle_data(file: Path):
    with file.open(mode="rb") as pickle_file:
        return pickle.load(pickle_file)


class X19MortalityDataset(Dataset):
    def __init__(
            self,
            x19_pickle: Path = pc.GOOD_PICKLE_DIR / pc.X19_PICKLE_FILENAME,
            y_pickle: Path = pc.GOOD_PICKLE_DIR / pc.Y_PICKLE_FILENAME
    ):
        self.x19 = torch.tensor(get_pickle_data(x19_pickle))
        # pickle file has many targets
        all_y = get_pickle_data(y_pickle)
        self.mort = torch.tensor([icu_stay[0] for icu_stay in all_y])

    def __len__(self):
        return len(self.mort)

    def __getitem__(self, idx: int):
        return self.x19[idx, :, :], self.mort[idx]
