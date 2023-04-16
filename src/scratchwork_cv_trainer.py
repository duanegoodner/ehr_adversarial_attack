import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import KFold
import numpy as np
import torch.nn as nn
from lstm_model import BinaryBidirectionalLSTM
from x19_mort_dataset import X19MortalityDataset


# Create your dataset object and get the number of samples
dataset = X19MortalityDataset()
num_samples = len(dataset)

model = BinaryBidirectionalLSTM(
    input_size=48, lstm_hidden_size=128, fc_hidden_size=32
)

# Define your batch size
batch_size = 128

# Define the number of folds for cross-validation
num_folds = 5

# Use KFold to create indices for each fold
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

for fold, (train_indices, validation_indices) in enumerate(
    kf.split(range(num_samples))
):
    # Create samplers for the training and validation sets
    # train_weights = np.zeros(num_samples)
    labels = np.array([dataset[i][1].item() for i in range(num_samples)])
    class_counts = np.bincount(labels)
    num_minority = class_counts[1]
    num_majority = class_counts[0]
    train_weights = np.where(labels == 0, 1 / num_majority, 1 / num_minority)
    # train_weights[labels == 0] = 1 / num_majority
    # train_weights[labels == 1] = 1 / num_minority
    train_sampler = WeightedRandomSampler(
        train_weights[train_indices], len(train_indices), replacement=True
    )
    # Create data loaders using the samplers
    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler
    )
    validation_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # Train your model for this fold
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Train your model with each batch
        print(batch_idx)

    # Evaluate your model on the validation set for this fold
    # model.eval()
    # for batch_idx, (data, target) in enumerate(validation_loader):
    #     # Evaluate your model with each batch
    #     pass

    # Compute the accuracy or other metrics for this fold
    # accuracy = ...
    # print(f"Fold {fold}: Accuracy = {accuracy}")