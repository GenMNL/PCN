import torch
from torch.utils.data import Dataset
from torch.utils.data import dataloader
import numpy as np
import open3d as o3d
import json
import os

class MakeDataset(Dataset):
    def __init__(self, dataset_path, subset, eval, num_partial_pattern, transform=None):
        super(MakeDataset, self).__init__()
        self.dataset_path = dataset_path # path of dataset
        self.subset = subset # The object which wants to train
        self.eval = eval # you can select train, test or validation
        self.num_partial_pattern = num_partial_pattern # number of pattern
        self.transform = transform # I don't define prepocessing
        self.ext = ".pcd" # the extension of point cloud data

    def __len__(self):
        return len(self.len_data)

    def __getitem__(self, index):
        # read json file
        read_json = open(f"{self.dataset_path}/PCN.json", "r")
        data_list = json.load(read_json)

        # get the id and index of object which wants to train(or test)
        for i in range(len(data_list)):
            dict_i = data_list[i]
            taxonomy_name = dict_i["taxonomy_name"]
            if taxonomy_name == self.subset:
                subset_index = i
                subset_id = dict_i["taxonomy_id"]
                break

        # make dataset path of completion point cloud.
        '''
        the length of completion dataset has to match with partial point cloud dataset.
        so you need to expand the array of dataet.
        In this case, I expand the path array of complete point cloud dataet.
        '''
        data_comp_list = data_list[subset_index][self.eval]
        data_comp_list = np.array(data_comp_list, dtype=str)
        data_comp_list = np.repeat(data_comp_list, self.num_partial_pattern)

        data_comp_path = os.path.join(self.dataset_path, "ShapeNetCompletion", self.eval, "complete", subset_id)
        data_comp_path = os.path.join(data_comp_path, data_comp_list[index]+self.ext)
        self.len_data = data_comp_path

        # make dataset path of partial point cloud
        partial_dir = data_list[subset_index][self.eval]
        data_partial_list = []
        for i in range(len(partial_dir)):
            for j in range(self.num_partial_pattern):
                data_partial_list.append(f"{partial_dir[i]}/0{j}")

        data_partial_path = os.path.join(self.dataset_path, "ShapeNetCompletion", self.eval, "partial", subset_id)
        data_partial_path = os.path.join(data_partial_path, data_partial_list[index]+self.ext)

        # get tensor from path
        # completion point cloud
        comp_pc = o3d.io.read_point_cloud(data_comp_path)
        comp_pc = np.asarray(comp_pc)
        comp_pc = torch.tensor(comp_pc)

        # partial point cloud
        partial_pc = o3d.io.read_point_cloud(data_partial_path)
        partial_pc = np.asarray(partial_pc)
        partial_pc = torch.tensor(partial_pc)

        return comp_pc, partial_pc
        # return data_comp_path , data_partial_path

if __name__ == "__main__":
    pc_dataset = MakeDataset("./data", "airplane", "test", 8)
    print(pc_dataset[0])