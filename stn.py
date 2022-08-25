import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from module import *

class STNkd(nn.Module):
    def __init__(self, num_channels, num_points, device):
        super(STNkd, self).__init__()
        self.num_channels = num_channels
        self.num_points = num_points
        self.device = device

        self.Conv_ReLU = nn.Sequential(
            nn.Conv1d(self.num_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.MaxPool = nn.Sequential(
            MaxPooling(1024, self.num_points)
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.num_channels**2)
        )

    def forward(self, input):
        batchsize = input.shape[0]

        x = self.Conv_ReLU(input)
        x = self.MaxPool(x)
        x = self.fc(x)

        iden = np.eye(self.num_channels).flatten().astype(np.float32)
        iden = Variable(torch.from_numpy(iden)).view(1, self.num_channels**2).repeat(batchsize, 1)
        iden = iden.to(self.device)

        out = x + iden
        out = out.view(-1, self.num_channels, self.num_channels)

        return out

if __name__ == "__main__":
    input = torch.randn(10, 3, 200)
    stn = STNkd(3, 200, "cpu")
    out = stn(input)
    print(out.shape)
