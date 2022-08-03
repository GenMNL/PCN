import torch
import torch.nn as nn
import torch.nn.functional as F
from module import *

# ----------------------------------------------------------------------------------------
# PCN uses PointNet for encoder 
class PointNet(nn.Module):
    def __init__(self, num_points, emb_dim, device):
        super(PointNet, self).__init__()
        self.num_points = num_points
        self.emb_dim = emb_dim
        self.device = device

        # MLP1 use for getting point feature
        self.MLP1 = nn.Sequential(
            Conv_ReLU(3, 128), # Conv_ReLU is original module
            Conv_ReLU(128, 256)
        )
        # MLP2 use for getting point feature which concern global and point feature
        self.MLP2 = nn.Sequential(
            Conv_ReLU(256*2, 512),
            Conv_ReLU(512, self.emb_dim)
        )
        # MaxPool1 use for getting global feature
        self.MaxPool1 = nn.Sequential(
            MaxPooling(256, self.num_points) # self.num_points is used for view of tensor
        )
        # MaxPool2 use for getting encoder result
        self.MaxPool2 = nn.Sequential(
            MaxPooling(self.emb_dim, self.num_points) # self.num_points is used for view of tensor
        )


    def forward(self, input_data):
        input_data = input_data.permute(0, 2, 1) # torchのconv1dでは，真ん中をチャンネルとして畳みこむためpermuteで並び変え
        point_feature1 = self.MLP1(input_data) # point feature only concern local feature
        global_feature1 = self.MaxPool1(point_feature1) # get global feature
        global_feature1 = torch.unsqueeze(global_feature1, 2) # unsqueezeはサイズ１の次元をテンソルに追加する

        # concatenate global and point feature
        x = global_feature1.repeat(1, 1, self.num_points) # repeat self.num_points times ind direction of dim=2
        x = torch.cat([x, point_feature1], dim=1) # concatenate point feature and global feature tensors.

        # apply MLP and MaxPool for new concatenated tensor and get global feature
        point_feature2 = self.MLP2(x) # x = (batchsize, channel, num_points)
        out = self.MaxPool2(point_feature2)

        return out

# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# test
if __name__ == "__main__":
    input = torch.randn(1, 200, 3, device="cuda") # (bachsize, num_point, channnel)
    pointnet= PointNet(200, 1024, "cuda").to("cuda") # 2000 is num of points
    test_coarse_output = pointnet(input)
    print(test_coarse_output.device)