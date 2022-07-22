import torch
import torch.nn as nn

tensor = torch.randn(2, 5)
print(tensor)
linear = nn.Linear(5, 10)
out = linear(tensor)
print(out.shape)