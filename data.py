import torch
from torch.utils.data import Dataset
from torch.utils.data import dataloader
import numpy as np
import open3d as o3d
import json
import os

# ----------------------------------------------------------------------------------------
# make collate function for dataloader
class OriginalCollate():
    def __init__(self, num_partial, num_comp, device):
        self.num_partial = num_partial
        self.num_comp = num_comp
        self.device = device

    def __call__(self, batch_list):
        # get batch size
        batch_size = len(batch_list)

        # * in *batch_list is transpose of batch_list
        # There are as many tensors as there are batchsize in batch_list
        # comp_batch and partial_batch are tuple which include many tensors
        comp_batch, partial_batch, a, b, c, d = list(zip(*batch_list))
        # transform tuple of complete point cloud to tensor
        comp_batch = list(comp_batch)
        for i in range(batch_size):
            n = len(comp_batch[i])
            idx = np.random.permutation(n)
            if len(idx) < self.num_comp:
                temp = np.random.randint(0, n, size=(self.num_comp- n))
                idx = np.concatenate([idx, temp])
            comp_batch[i] = comp_batch[i][idx[:self.num_comp], :]

        # torch.stack concatenate each tensors in the direction of the specified dim(dim=0)
        comp_batch = torch.stack(comp_batch, dim=0).to(self.device)

        # transform tuple of partial point cloud to tensor
        # num of point in each tensor of partial point cloud change to the same num
        partial_batch = list(partial_batch)
        for i in range(batch_size):
            n = len(partial_batch[i])
            idx = np.random.permutation(n)
            if len(idx) < self.num_partial:
                temp = np.random.randint(0, n, size=(self.num_partial - n))
                idx = np.concatenate([idx, temp])
            partial_batch[i] = partial_batch[i][idx[:self.num_partial], :]

        partial_batch = torch.stack(partial_batch, dim=0).to(self.device)

        a = np.array(list(a))
        b = np.array(list(b))
        c = np.array(list(c))
        d = np.array(list(d))
        # There are tensor which is board on args.device(default is cuda).
        return comp_batch, partial_batch, a, b, c, d
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# make collate function for dataloader
class DataNormalization():
    def __init__(self):
        pass

    def __call__(self, ary):
        max = ary.max(axis=0)
        min = ary.min(axis=0)
        x_max, y_max, z_max = max[0], max[1], max[2]
        x_min, y_min, z_min = min[0], min[1], min[2]

        ary[:,0] -= x_min
        ary[:,1] -= y_min
        ary[:,2] -= z_min
        ary[:,0] /= (x_max - x_min)
        ary[:,1] /= (y_max - y_min)
        ary[:,2] /= (z_max - z_min)

        return ary, max, min
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# make class for dataset
class MakeDataset(Dataset):
    def __init__(self, dataset_path, subset, eval, num_partial_pattern, device, transform=DataNormalization):
        super(MakeDataset, self).__init__()
        self.dataset_path = dataset_path # path of dataset
        self.subset = subset # The object which wants to train
        self.eval = eval # you can select train, test or validation
        self.num_partial_pattern = num_partial_pattern # number of pattern
        self.device = device
        self.transform = transform() # min-max normalization of input array
        self.ext = ".pcd" # the extension of point cloud data

    def __len__(self):
        subset_index, _ = self.get_item_from_json()
        data_comp_list = self.data_list[subset_index][self.eval]
        data_comp_list = np.array(data_comp_list, dtype=str)

        if self.num_partial_pattern != 0: # when there are some patterns of partial data, repeat array.
            data_comp_list = np.repeat(data_comp_list, self.num_partial_pattern)
        len_data = len(data_comp_list) # make instance variable for count length of data.

        return len_data

    def __getitem__(self, index):
        subset_index, subset_id = self.get_item_from_json()

        # ///
        # make dataset path of completion point cloud.
        '''
        the length of completion dataset has to match with partial point cloud dataset.
        so you need to expand the array of dataet.
        In this case, I expand the path array of complete point cloud dataet.
        '''
        data_comp_list = self.data_list[subset_index][self.eval]
        data_comp_list = np.array(data_comp_list, dtype=str)

        if self.num_partial_pattern != 0: # when there are some patterns of partial data, repeat array.
            data_comp_list = np.repeat(data_comp_list, self.num_partial_pattern)

        data_comp_path = os.path.join(self.dataset_path, self.eval, "complete", subset_id)
        data_comp_path = os.path.join(data_comp_path, data_comp_list[index]+self.ext)

        # make dataset path of partial point cloud
        partial_dir = self.data_list[subset_index][self.eval]
        data_partial_list = []
        for i in range(len(partial_dir)):
            if self.num_partial_pattern != 0:
                for j in range(self.num_partial_pattern):
                    data_partial_list.append(f"{partial_dir[i]}/0{j}")
            else:
                data_partial_list.append(f"{partial_dir[i]}/00")

        data_partial_path = os.path.join(self.dataset_path, self.eval, "partial", subset_id)
        data_partial_path = os.path.join(data_partial_path, data_partial_list[index]+self.ext)

        # ///
        # get tensor from path
        # completion point cloud
        comp_pc = o3d.io.read_point_cloud(data_comp_path)
        comp_pc_visu = comp_pc # if you want to visualize data, input this to open3d.visualization
        comp_pc = np.asarray(comp_pc.points)
        comp_pc, comp_max, comp_min = self.transform(comp_pc)
        comp_pc = torch.tensor(comp_pc, dtype=torch.float, device=self.device)

        # partial point cloud
        partial_pc = o3d.io.read_point_cloud(data_partial_path)
        partial_pc_visu = partial_pc # if you want to visualize data, input this to open3d.visualization
        partial_pc = np.asarray(partial_pc.points)
        partial_pc, partial_max, partial_min = self.transform(partial_pc)
        partial_pc = torch.tensor(partial_pc, dtype=torch.float, device=self.device)

        return comp_pc, partial_pc, comp_max, comp_min, partial_max, partial_min
        # return comp_pc, partial_pc, comp_pc_visu, partial_pc_visu # use this if you want to visualize point cloud
        # return data_comp_path , data_partial_path # use this if you want to check in your pc which don't have cuda.

    def get_item_from_json(self):
        # read json file
        read_json = open(f"{self.dataset_path}/PCN.json", "r")
        self.data_list = json.load(read_json)

        # get the id and index of object which wants to train(or test)
        for i in range(len(self.data_list)):
            dict_i = self.data_list[i]
            taxonomy_name = dict_i["taxonomy_name"]
            if taxonomy_name == self.subset:
                subset_index = i
                subset_id = dict_i["taxonomy_id"]
                break

        return subset_index, subset_id
# ----------------------------------------------------------------------------------------

if __name__ == "__main__":
    pc_dataset = MakeDataset("./data/BridgeCompletion", "bridge", "train", 4, "cuda")
    # i = 46000
    i = 2
    print(pc_dataset[0][i].min())
    print(pc_dataset[0][i].max())
    print(pc_dataset[10][i].shape)

    # o3d.visualization.draw_geometries([pc_dataset[7][3]])
