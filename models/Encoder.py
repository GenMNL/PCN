import torch
import torch.nn as nn
import torch.nn.functional as F
from module import Conv_ReLU
from stn import STNkd

# ----------------------------------------------------------------------------------------
# PCN uses PointNet for encoder 
class PointNet(nn.Module):
    def __init__(self, emb_dim):
        super(PointNet, self).__init__()
        self.emb_dim = emb_dim

        # initial spational transformed network
        self.STN3d = nn.Sequential(
            STNkd(3, self.num_points)
        )
        # MLP1 use for getting point feature
        self.MLP1 = nn.Sequential(
            Conv_ReLU(3, 128), # Conv_ReLU is original module
            Conv_ReLU(128, 256)
        )
        # second spational transformed network
        self.STN256d = nn.Sequential(
            STNkd(256, self.num_points)
        )
        # MLP2 use for getting point feature which concern global and point feature
        self.MLP2 = nn.Sequential(
            Conv_ReLU(256*2, 512),
            Conv_ReLU(512, self.emb_dim)
        )

    def forward(self, input_data):
        input_data = input_data.permute(0, 2, 1) # torchのconv1dでは，真ん中をチャンネルとして畳みこむためpermuteで並び変え
        _, _, N = input_data.shape

        # apply first stn
        # trans_3d = self.STN3d(input_data)
        # x = input_data.permute(0, 2, 1)
        # trans_input = torch.bmm(x, trans_3d)
        # trans_input = trans_input.permute(0, 2, 1)

        # point_feature1 = self.MLP1(trans_input) # point feature only concern local feature
        point_feature1 = self.MLP1(input_data) # point feature only concern local feature

        # apply second stn
        # trans_256d = self.STN256d(point_feature1)
        # x = point_feature1.permute(0, 2, 1)
        # trans_point_feature1 = torch.bmm(x, trans_256d)
        # trans_point_feature1 = trans_point_feature1.permute(0, 2, 1)

        # global_feature1 = self.MaxPool1(trans_point_feature1) # get global feature
        global_feature1 = torch.max(point_feature1, dim=2, keepdim=True)[0]

        # concatenate global and point feature
        x = global_feature1.repeat(1, 1, N) # repeat self.num_points times ind direction of dim=2
        # x = torch.cat([x, trans_point_feature1], dim=1) # concatenate point feature and global feature tensors.
        x = torch.cat([x, point_feature1], dim=1) # concatenate point feature and global feature tensors.

        # apply MLP and MaxPool for new concatenated tensor and get global feature
        point_feature2 = self.MLP2(x) # x = (batchsize, channel, num_points)
        out = torch.max(point_feature2, dim=2)[0]

        return out

# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# test
if __name__ == "__main__":
    input = torch.randn(3, 200, 3, device="cuda") # (bachsize, num_point, channnel)
    pointnet= PointNet(200, 1024, "cuda").to("cuda") # 2000 is num of points
    test_coarse_output = pointnet(input)
    print(test_coarse_output.device)
