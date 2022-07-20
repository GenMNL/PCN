import torch
import torch.nn as nn
import torch.nn.functional as F
from module import *

# ----------------------------------------------------------------------------------------
# PCN uses PointNet for encoder 
class PointNet(nn.Module):
    def __init__(self, num_points):
        super(PointNet, self).__init__()
        self.num_points = num_points

        # MLP1 use for getting point feature
        self.MLP1 = nn.Sequential(
            Conv_ReLU(3, 128, 1), # Conv_ReLU is original module
            Conv_ReLU(128, 256)
        )
        # MLP2 use for getting point feature which concern global and point feature
        self.MLP2 = nn.Sequential(
            Conv_ReLU(256*2, 512, 1),
            Conv_ReLU(512, 1024, 1)
        )
        # MaxPool1 use for getting global feature
        self.MaxPool1 = nn.Sequential(
            MaxPooling(256, self.num_points)
        )
        # MaxPool2 use for getting encoder result
        self.MaxPool2 = nn.Sequential(
            MaxPooling(1024, self.num_points)
        )


    def forward(self, input_data):
        point_feature1 = self.MLP1(input_data) # point feature only concern local feature
        global_feature1 = self.MaxPool1(point_feature1) # get global feature

        # concatenate global and point feature
        x = global_feature1.repeat(self.num_points, 1)
        x = nn.cat([x, point_feature1], dim=1)

        # apply MLP and MaxPool for new concatenated tensor and get global feature
        point_feature2 = self.MLP2(x)
        out = self.MaxPool2(point_feature2)

        return out

# ----------------------------------------------------------------------------------------
