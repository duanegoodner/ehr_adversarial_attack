import pickle
import torch
from pathlib import Path
import project_config as pc


def get_pickle_data(file: Path):
    with file.open(mode="rb") as pickle_file:
        return pickle.load(pickle_file)


x19_orig_obj = get_pickle_data(pc.GOOD_PICKLE_DIR / pc.X19_PICKLE_FILENAME)

