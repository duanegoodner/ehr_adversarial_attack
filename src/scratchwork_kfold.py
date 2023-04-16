import numpy as np
from sklearn.model_selection import KFold
from x19_mort_dataset import X19MortalityDataset

dataset = X19MortalityDataset()
num_folds = 5

kf = KFold(n_splits=num_folds, shuffle=True)

# since random_state arg not given to KFold constructor, each call
# to KFold.split() gives different split indices

for fold_idx, (train_indices, validation_indices) in enumerate(
        kf.split(range(len(dataset)))):
    print("checking split")

for fold_idx, (train_indices, validation_indices) in enumerate(
        kf.split(range(len(dataset)))):
    print("checking split")

