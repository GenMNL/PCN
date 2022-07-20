import numpy as np
import torch
import torch.nn as nn

tensor = torch.randn(1, 5, 3)
print(tensor)

tensor = tensor.view(-1, 3, 5)
print(tensor)

mp = nn.MaxPool1d(5)
out = mp(tensor)
print(out)
