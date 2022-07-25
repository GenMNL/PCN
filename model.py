import torch
import torch.nn as nn
import Encoder
import Decoder

# ----------------------------------------------------------------------------------------
class PCN(nn.Module):
    def __init__(self, num_points, emb_dim, num_coarse, grid_size):
        super(PCN, self).__init__()
        self.num_points = num_points
        self.emb_dim = emb_dim
        self.num_coarse = num_coarse
        self.grid_size = grid_size
        self.Encoder = Encoder.PointNet(num_points=self.num_points, emb_dim=self.emb_dim)
        self.Decoder_coarse= Decoder.AffineDecoder(num_coarse=self.num_coarse, emb_dim=self.emb_dim)
        self.Decoder_fine = Decoder.FineDecoder(grid_size=self.grid_size, num_coarse=self.num_coarse, emb_dim=self.emb_dim)

    def forward(self, input_data):
        feature_v = self.Encoder(input_data)
        coarse_out = self.Decoder_coarse(feature_v)
        result = self.Decoder_fine(coarse_out, feature_v)

        return feature_v, coarse_out, result
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# test
if __name__ == "__main__":
    input = torch.randn(10, 2000, 3)
    model = PCN(2000, 1024, 1024, 4)
    out_feature, out_coarse, out_result = model(input)
    print(out_feature.shape)
    print(out_coarse.shape)
    print(out_result.shape)