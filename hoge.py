import torch
import torch.nn as nn
import numpy as np
import torch

n = 20
pc = torch.randn(3, 3)
print(pc)
print(pc[0, :])
idx = np.random.permutation(pc.shape[0])
idx = np.concatenate([idx, np.random.randint(pc.shape[0], size=n-pc.shape[0])])
out = pc[idx[:n], :]
print(out.shape)

# arr = np.arange(3, 6)
# print(arr[[0, 1, 0, 2]])
# print(arr)