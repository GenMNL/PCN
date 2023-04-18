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
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU()) # nn.ReLU()はsequentialに入れられるが，F.reluは入れられない．（厳密にいうともう少しあるかも）

    def forward(self, input_data):
        out = self.main(input_data)
        return out
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# test
if __name__ == "__main__":
    input = torch.randn(10, 3, 2000, device="cuda")

    # conv = Conv_ReLU(3, 128).to('cuda')
    conv = Conv_ReLU(3, 128)
    out_conv = conv(input)

    # MaxPool = MaxPooling(3, 2000).to("cuda")
    out_mp = torch.max(out_conv, dim=2)[0]

    print(out_mp.device)
    print(out_conv.device)
