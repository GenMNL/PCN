import torch
import torch.nn as nn
import numpy as np
import torch
from pytorch3d.loss import chamfer_distance

perm = np.random.permutation(100)
print(perm)