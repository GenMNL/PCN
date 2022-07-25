import torch
import torch.nn as nn
import Encoder
import Decoder

# ----------------------------------------------------------------------------------------
class PCN(nn.Module):
    def __init__(self, num_points, num_coarse, grid_size):
        super(PCN, self).__init__()
        self.num_points = num_points
        self.num_coarse = num_coarse
        self.grid_size = grid_size
        self.Encoder = Encoder.PointNet(self.num_points)
        self.Decoder_coarse= Decoder.AffineDecoder(self.num_coarse)
        self.Decoder_fine = Decoder.FineDecoder(self.grid_size, self.num_coarse)

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
    model = PCN(2000, 1024, 4)
    out_feature, out_coarse, out_result = model(input)
    print(out_feature.shape)
    print(out_coarse.shape)
    print(out_result.shape)