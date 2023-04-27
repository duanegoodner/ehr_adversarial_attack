import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


# Example usage
my_data = [1, 2, (3, 3, 3), 4, 5]
my_dataset = MyDataset(my_data)

# Retrieve the item at index 2
item = my_dataset[2]
print(item)  # Output: 3
