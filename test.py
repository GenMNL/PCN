import torch
from pytorch3d.loss import chamfer_distance

point1 = torch.randn(10, 100, 3).cuda()
point2 = torch.randn(10, 50, 3).cuda()

cd, _ = chamfer_distance(point1, point2)
print(cd)