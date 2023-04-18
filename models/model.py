import torch
import torch.nn as nn
from models.Encoder import PointNet
from models.Decoder import AffineDecoder, FineDecoder

# ----------------------------------------------------------------------------------------
class PCN(nn.Module):
    def __init__(self, emb_dim, num_coarse, grid_size, device):
        super(PCN, self).__init__()
        self.Encoder = PointNet(emb_dim).to(device)
        self.Decoder_coarse = AffineDecoder(num_coarse, emb_dim).to(device)
        self.Decoder_fine = FineDecoder(grid_size, num_coarse, emb_dim).to(device)

    def forward(self, input_data):
        feature_v = self.Encoder(input_data)
        coarse_out = self.Decoder_coarse(feature_v)
        result = self.Decoder_fine(coarse_out, feature_v)

        return feature_v, coarse_out, result
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# test
if __name__ == "__main__":
    input = torch.randn(10, 2000, 3, device="cuda")
    model = PCN(2000, 1024, 1024, 4, "cuda").to("cuda")
    out_feature, out_coarse, out_result = model(input)
    print(out_feature.shape)
    print(out_coarse.shape)
    print(out_result.shape)
    print(out_feature.device)
    print(out_coarse.device)
    print(out_result.device)
