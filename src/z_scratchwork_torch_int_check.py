import torch

def is_int_tensor(x):
    return isinstance(x, torch.Tensor) and issubclass(x.dtype.type, torch.Integral)

# Example usage
x = torch.tensor([1, 2, 3], dtype=torch.int16)
print(is_int_tensor(x))  # Output: True

y = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
print(is_int_tensor(y))  # Output: False
