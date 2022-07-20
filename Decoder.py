import torch
import torch.nn as nn
import torch.nn.functional as F
from PointNet import *

class AffineDecoder(nn.Module):
    def __init__(self, num_coarse):
        super(AffineDecoder, self).__init__()
        self.num_coarse = num_coarse

        self.MakeCoarse = nn.Sequential(
            nn.Linear()
        )