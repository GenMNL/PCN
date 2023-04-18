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
        # read json file
        read_json = open(f"{self.dataset_path}/PCN.json", "r")
        data_list = json.load(read_json)

        if self.subset == "all":
            count = 0
            for i in range(len(data_list)):
                dict_i = data_list[i]
                num_each_subset = len(dict_i[self.eval])
                count += num_each_subset

        else:
            for i in range(len(data_list)):
                dict_i = data_list[i]
                taxonomy_name = dict_i["taxonomy_name"]
                if taxonomy_name == self.subset:
                    subset_id_index = i
                    break
            count = len(data_list[subset_id_index][self.eval])

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

        if self.subset == "all":
            data_comp_path, data_partial_path = self.get_path_from_all_subset(index)
        else:
            data_comp_path, data_partial_path = self.get_path_from_one_subset(index)

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

    def get_path_from_all_subset(self, index):

        path_list = []
        subset_id_list = []
        for i in range(len(self.data_list)):
            dict_i = self.data_list[i]
            path_list_each_subset = dict_i[self.eval]
            path_list.extend(path_list_each_subset)

            each_subset_id = np.array(dict_i["taxonomy_id"])
            each_subset_id = np.repeat(each_subset_id, len(path_list_each_subset))
            subset_id_list.extend(list(each_subset_id))

        subset_id_list = np.repeat(subset_id_list, self.num_partial_pattern)
        subset_id = subset_id_list[index]

        # make dataset path of complete point cloud
        data_comp_list = np.array(path_list, dtype=str)
        data_comp_list = np.repeat(data_comp_list, self.num_partial_pattern)
        data_comp_path = os.path.join(self.dataset_path, self.eval, "complete")
        data_comp_path = os.path.join(data_comp_path, subset_id, data_comp_list[index]+".pcd")

        # make dataset path of partial point cloud
        data_partial_list = []
        for i in range(len(path_list)):
            for j in range(self.num_partial_pattern):
                data_partial_list.append(f"{path_list[i]}/0{j}")
        data_partial_path = os.path.join(self.dataset_path, self.eval, "partial")
        data_partial_path = os.path.join(data_partial_path, subset_id, data_partial_list[index]+".pcd")

        return data_comp_path, data_partial_path

    def get_path_from_one_subset(self, index):
        # get the id and index of object which wants to train(or test)
        for i in range(len(self.data_list)):
            dict_i = self.data_list[i]
            taxonomy_name = dict_i["taxonomy_name"]
            if taxonomy_name == self.subset:
                subset_id_index = i
                subset_id = dict_i["taxonomy_id"]
                break

        path_list = self.data_list[subset_id_index][self.eval]

        # make dataset path of complete point cloud
        data_comp_list = np.array(path_list, dtype=str)
        data_comp_list = np.repeat(data_comp_list, self.num_partial_pattern)
        data_comp_path = os.path.join(self.dataset_path, self.eval, "complete", subset_id)
        data_comp_path = os.path.join(data_comp_path, data_comp_list[index]+".pcd")

        # make dataset path of partial point cloud
        data_partial_list = []
        for i in range(len(path_list)):
            for j in range(self.num_partial_pattern):
                data_partial_list.append(f"{path_list[i]}/0{j}")
        data_partial_path = os.path.join(self.dataset_path, self.eval, "partial", subset_id)
        data_partial_path = os.path.join(data_partial_path, data_partial_list[index]+".pcd")

        return data_comp_path, data_partial_path

# ----------------------------------------------------------------------------------------

if __name__ == "__main__":
    pc_dataset = MakeDataset("./data/ShapeNetCompletion", "chair", "train", 1, "cuda")
    i = 46000
    # i = 1
    print(len(pc_dataset))
    print(pc_dataset[205])

    # o3d.visualization.draw_geometries([pc_dataset[7][3]])
