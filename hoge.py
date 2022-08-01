import torch
import numpy as np
import torch

tensor = torch.randn(10,3)
print(tensor)
a = np.arange(10)
a = np.random.permutation(a)
print(a)
tensor = tensor[a[0:3], :]
print(tensor)