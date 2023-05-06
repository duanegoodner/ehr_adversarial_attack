import numpy as np

data = np.array([[1, 2, 3], [1, 3, 4], [2, 4, 5], [2, 5, 6]])
groups, indices = np.unique(data[:, 0], return_inverse=True)
grouped_data = [data[indices == i] for i in range(len(groups))]
