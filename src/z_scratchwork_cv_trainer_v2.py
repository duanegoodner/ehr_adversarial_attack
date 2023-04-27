from x19_mort_dataset import X19MortalityDataset
from torch.utils.data import WeightedRandomSampler, DataLoader, Subset
import numpy as np
import torch
from sklearn.model_selection import KFold

dataset = X19MortalityDataset()
dataset_size = dataset.__len__()
num_folds = 5
batch_size = 128
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)


for fold_idx, (train_indices, validation_indices) in enumerate(
    kf.split(range(dataset_size))
):
    train_dataset = Subset(dataset=dataset, indices=train_indices)
    y_train_indices = train_dataset.indices
    y_train = [dataset[i][1] for i in y_train_indices]
    class_sample_counts = np.unique(y_train, return_counts=True)[1]
    weight = 1.0 / class_sample_counts
    sample_weights = torch.from_numpy(np.array([weight[t] for t in y_train]))
    train_sampler = WeightedRandomSampler(
        sample_weights.type("torch.DoubleTensor"), len(sample_weights)
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler
    )
    for batch_idx, (data, target) in enumerate(train_dataloader):
        print(batch_idx)

    validation_dataset = Subset(
        dataset=dataset, indices=validation_indices
    )
    validation_dataloader = DataLoader(
        dataset=validation_dataset, batch_size=batch_size, shuffle=True
    )
    for batch_idx, (data, target) in enumerate(validation_dataloader):
        print(batch_idx)
