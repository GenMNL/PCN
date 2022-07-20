import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------------------
# original module (nn.ReLU(nn.conv1d()) -> nn.BatchNorm1d) using in PointNet.py
class Conv_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_ReLU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.main= nn.Sequential(
            nn.Conv1d(self.in_channels, self.out_channels, 1),
            nn.ReLU(), # nn.ReLU()はsequentialに入れられるが，F.reluは入れられない．（厳密にいうともう少しあるかも）
            nn.BatchNorm1d(self.out_channels)) # 入力の点群にNNをかけたものを正規化している（tensor全体の平均0の分散1）．(1,3,5) -> (-1.2, 0, 1.2)

    def forward(self, input_data):
        out = self.main(input_data)
        return out
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# max pooling for PointNet 
class MaxPooling(nn.Module):
    def __init__(self, num_channels, num_points):
        super(MaxPooling, self).__init__()
        self.num_channels = num_channels
        self.num_points = num_points
        self.main = nn.MaxPool1d(self.num_points)

    def forward(self, input_data):
        x = input_data.view(-1, self.num_channels, self.num_points)
        x = self.main(x)
        out = x.view(-1, self.num_channels)

        return out
# ----------------------------------------------------------------------------------------