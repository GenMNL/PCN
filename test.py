# import open3d as o3d
# import os
# import numpy as np
# import numpy as np
# from torch.utils.data import Dataset
# import torch

# # pcd = o3d.io.read_point_cloud('./data/1a04e3eab45ca15dd86060f189eb133.pcd')
# # print(type(pcd))
# # print(np.asarray(pcd.points))
# # o3d.visualization.draw_geometries([pcd])

# class MakeDataset(Dataset):
#     def __init__(self):
#         super(MakeDataset, self).__init__()
#         self.data = []
    
#     def __len__(self):
#         return len(self.path)

#     def __getitem__(self, index):
#         path1 = os.path.join('./data/1a04e3eab45ca15dd86060f189eb133.pcd')
#         path2 = os.path.join('./data/cc113b6e9d4fbeb23df325aac2f73830.pcd')
    
#         self.path = [path1, path2]
#         path_item = self.path[index]

#         tensor = o3d.io.read_point_cloud(path_item)
#         tensor = np.asarray(tensor.points)
#         tensor = torch.tensor(tensor)

#         return tensor

# if __name__ == '__main__':
#     tensor = MakeDataset()
#     print(tensor[0])

import json
import os
import numpy as np

par_path = [i for i in range(8)]
par_path = np.array(par_path, dtype=str)
# path_list = []
# for j in range(len(par_path)):
#     for i in range(10, 18):
#         path_list.append(f"{par_path[j]}/{i}")

# print(path_list)

hoge = [f'{par_path[j]}/{i}' for j in range(len(par_path)) for i in range(10, 18)]
print(hoge)

for j in range(len(par_path)):
    for i in range(10, 18):
        par_path[j] = par_path[j]+f'/0{i}'

print(par_path)