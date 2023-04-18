import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class STNkd(nn.Module):
    def __init__(self, num_channels):
        super(STNkd, self).__init__()
        self.num_channels = num_channels

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
        device = input.device

        batchsize = input.shape[0]

        x = self.Conv_ReLU(input)
        x = torch.max(x, dim=2)[0]
        x = self.fc(x)

        iden = np.eye(self.num_channels).flatten().astype(np.float32)
        iden = Variable(torch.from_numpy(iden)).view(1, self.num_channels**2).repeat(batchsize, 1)
        iden = iden.to(device)

        out = x + iden
        out = out.view(-1, self.num_channels, self.num_channels)

        return out

if __name__ == "__main__":
    input = torch.randn(10, 3, 200)
    stn = STNkd(3, 200, "cpu")
    out = stn(input)
    print(out.shape)
