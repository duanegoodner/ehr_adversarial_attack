import numpy as np


def write_to_array(records_array: list, idx: int):
    records_array.append(np.random.randint(low=0, high=100, dtype=np.longlong))


my_array = []

for i in range(10):
    write_to_array(records_array=my_array, idx=i)


print(my_array)


