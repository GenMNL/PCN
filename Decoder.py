import torch
import torch.nn as nn
import torch.nn.functional as F
from Encoder import *
from module import *

# ----------------------------------------------------------------------------------------
# AffineDecoder is fully connection type decoder which generate coarse point cloud
class AffineDecoder(nn.Module):
    def __init__(self, num_coarse, emb_dim, device):
        super(AffineDecoder, self).__init__()
        self.num_coarse = num_coarse
        self.emb_dim = emb_dim
        self.device = device

        self.MakeCoarse = nn.Sequential(
            nn.Linear(self.emb_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.num_coarse*3)
        )

    def forward(self, global_feature_vector):
        out = self.MakeCoarse(global_feature_vector)
        out = torch.unsqueeze(out, dim=2)
        out = out.view(-1, self.num_coarse, 3)

        return out
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# FineDecoder is folding type decoder which make fine point cloud
class FineDecoder(nn.Module):
    def __init__(self, grid_size, num_coarse, emb_dim, device):
        super(FineDecoder, self).__init__()
        self.grid_size = grid_size
        self.num_coarse = num_coarse
        self.emb_dim = emb_dim
        self.device = device
        self.MLP = nn.Sequential(
            Conv_ReLU(self.emb_dim+2+3, 512),
            Conv_ReLU(512, 512),
            Conv_ReLU(512, 3),
        )

    def forward(self, coarse_output, global_feature):
        self.batchsize = coarse_output.shape[0]

        # make grid tensor
        grid_node = torch.linspace(-0.5, 0.5, steps=self.grid_size, device=self.device)
        grid = torch.meshgrid(grid_node, grid_node) # This is tuple object which contains x and y coordinates
        grid = torch.stack(grid, dim=2) # concatenate grid_x and grid_y in the direction of axis=2
        grid = grid.view(-1, 2) # make one of grid feature vector
        grid_feature = grid.repeat(self.num_coarse, 1)
        grid_feature = torch.unsqueeze(grid_feature, dim=0)
        grid_feature = grid_feature.repeat(self.batchsize, 1, 1)

        # expand coarse output tensor
        x = coarse_output.repeat(1, 1, self.grid_size**2)
        coarse_feature = x.view(self.batchsize, self.num_coarse*(self.grid_size**2), -1)

        # expand global feature tensor
        global_feature = torch.unsqueeze(global_feature, dim=1)
        global_feature = global_feature.repeat(1, self.num_coarse*(self.grid_size**2), 1)

        # concatenate all feature tensors
        features = torch.cat([global_feature, grid_feature, coarse_feature], dim=2)

        # adapt layer to features
        features = features.permute(0, 2, 1)
        fine_output = self.MLP(features)
        fine_output = fine_output + coarse_feature.permute(0, 2, 1)
        fine_output = fine_output.permute(0, 2, 1)

        return fine_output

# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# test
if __name__ == "__main__":
    input = torch.randn(10, 1024, device="cuda") # (batchsize, dim of feature vector)
    affine_decoder = AffineDecoder(num_coarse=1024, emb_dim=1024, device="cuda").to("cuda") # 1024 is the number of coarse point clouds.
    coarse_output = affine_decoder(input)

    fine_decoder = FineDecoder(4, num_coarse=1024, emb_dim=1024, device="cuda").to('cuda')
    fine_output = fine_decoder(coarse_output, input)
    print(fine_output.shape)
# ----------------------------------------------------------------------------------------
