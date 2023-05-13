import torch
from torch.utils.data import Dataset
import numpy as np
import open3d as o3d
import json
import os

# ----------------------------------------------------------------------------------------
# make collate function for dataloader
class OriginalCollate():
    def __init__(self, device):
        self.device = device

    def __call__(self, batch_list):
        # get batch size
        batch_size = len(batch_list)

        comp_batch, partial_batch, a, b, c, d = list(zip(*batch_list))

        # transform tuple of complete point cloud to tensor
        comp_batch = list(comp_batch)
        max_num_comp_in_batch = self.count_max_num_in_batch(batch_size, comp_batch)
        for i in range(batch_size):
            n = len(comp_batch[i])
            idx = np.random.permutation(n)
            if n < max_num_comp_in_batch:
                unique_idx = np.random.randint(0, n, size=(max_num_comp_in_batch - n))
                idx = np.concatenate([idx, unique_idx])
            comp_batch[i] = comp_batch[i][idx, :]
        comp_batch = torch.stack(comp_batch, dim=0).to(self.device)

        # transform tuple of partial point cloud to tensor
        partial_batch = list(partial_batch)
        max_num_partial_in_batch = self.count_max_num_in_batch(batch_size, partial_batch)
        for i in range(batch_size):
            n = len(partial_batch[i])
            idx = np.random.permutation(n)
            if n < max_num_partial_in_batch:
                unique_idx = np.random.randint(0, n, size=(max_num_partial_in_batch - n))
                idx = np.concatenate([idx, unique_idx])
            partial_batch[i] = partial_batch[i][idx, :]
        partial_batch = torch.stack(partial_batch, dim=0).to(self.device)

        a = np.array(list(a))
        b = np.array(list(b))
        c = np.array(list(c))
        d = np.array(list(d))
        # There are tensor which is board on args.device(default is cuda).
        return comp_batch, partial_batch, a, b, c, d
    
    def count_max_num_in_batch(self, batch_size, batch_list):
        # get max num points in each batch
        max_num_points = 0
        for j in range(batch_size):
            n = len(batch_list[j])
            if max_num_points < n:
                max_num_points = n
        return max_num_points
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# make collate function for dataloader
class DataNormalization():
    def __init__(self):
        pass

    def __call__(self, ary):
        max_value = ary.max(axis=0)
        min_value = ary.min(axis=0)

        normalized_ary = (ary - min_value)/(max_value - min_value)
        return normalized_ary, max_value, min_value
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

    def __len__(self):
        subset_id_index = 0

        # read json file
        read_json = open(f"{self.dataset_path}/PCN.json", "r")
        data_list = json.load(read_json)
        dict_each_bridge = data_list[subset_id_index][self.eval]

        count = 0
        for bridge_name in dict_each_bridge:
            each_bridge_component_list = dict_each_bridge[bridge_name]
            count += len(each_bridge_component_list)

        return count*self.num_partial_pattern

    def __getitem__(self, index):
        # ///
        # make dataset path of completion point cloud.
        '''
        the length of completion dataset has to match with partial point cloud dataset.
        so you need to expand the array of dataet.
        In this case, I expand the path array of complete point cloud dataet.
        '''
        # read json file
        read_json = open(f"{self.dataset_path}/PCN.json", "r")
        self.data_list = json.load(read_json)

        data_comp_path, data_partial_path = self.get_path(index)

        # ///
        # get tensor from path
        # completion point cloud
        comp_pc = o3d.io.read_point_cloud(data_comp_path)
        comp_pc = np.asarray(comp_pc.points)
        comp_pc, comp_max, comp_min = self.transform(comp_pc)
        comp_pc = torch.tensor(comp_pc, dtype=torch.float, device=self.device)

        # partial point cloud
        partial_pc = o3d.io.read_point_cloud(data_partial_path)
        partial_pc = np.asarray(partial_pc.points)
        partial_pc, partial_max, partial_min = self.transform(partial_pc)
        partial_pc = torch.tensor(partial_pc, dtype=torch.float, device=self.device)

        return comp_pc, partial_pc, comp_max, comp_min, partial_max, partial_min


    def get_path(self, index):
        # get the id and index of object which wants to train(or test)
        subset_id = "0000"
        subset_index = 0

        dict_each_bridge = self.data_list[subset_index][self.eval]

        comp_path_list = []
        partial_path_list = []
        for bridge_name in dict_each_bridge:
            component_list = dict_each_bridge[bridge_name]

            for i in range(self.num_partial_pattern):
                for component in component_list:
                    comp_component_path = os.path.join(bridge_name, component+".pcd")
                    partial_component_path = os.path.join(bridge_name, f"0{i}", component+".pcd")

                    comp_path_list.append(comp_component_path)
                    partial_path_list.append(partial_component_path)

        # make dataset path of complete point cloud
        all_comp_path_list = np.array(comp_path_list, dtype=str)
        comp_path_par_dir = os.path.join(self.dataset_path, self.eval, "complete", subset_id)
        all_comp_path_list = os.path.join(comp_path_par_dir, all_comp_path_list[index])

        # make dataset path of partial point cloud
        all_partial_path_list = np.array(partial_path_list, dtype=str)
        partial_path_par_dir = os.path.join(self.dataset_path, self.eval, "partial", subset_id)
        all_partial_path_list = os.path.join(partial_path_par_dir, all_partial_path_list[index])

        return all_comp_path_list, all_partial_path_list

# ----------------------------------------------------------------------------------------

if __name__ == "__main__":
    pc_dataset = MakeDataset("./", "bridge", "train", 3, "cpu")
    # i = 46000
    i = 1
    print(len(pc_dataset))
    print(pc_dataset[390])

    # o3d.visualization.draw_geometries([pc_dataset[7][3]])
