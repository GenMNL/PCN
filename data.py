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
        subset_index, subset_id = self.get_item_from_json()
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
        comp_pc = torch.tensor(comp_pc)

        # partial point cloud
        partial_pc = o3d.io.read_point_cloud(data_partial_path)
        partial_pc_visu = partial_pc # if you want to visualize data, input this to open3d.visualization
        partial_pc = np.asarray(partial_pc.points)
        partial_pc = torch.tensor(partial_pc)

        return comp_pc, partial_pc
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

class MakeDataloader(MakeDataset):
    def __init__(self, dataset_path, subset, eval, num_partial_pattern, transform,
                 batch_size, shuffle, drop_last):
        # for dataset 
        super().__init__(dataset_path, subset, eval, num_partial_pattern, transform)
        # for dataloader
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def original_collate(self, batch_list):
        # get batch size
        batch_size = np.array(batch_list).shape[0]

        # translate complete list to tensor.
        comp_batch, partial_batch = list(zip(*batch_list))
        comp_batch = torch.stack(comp_batch, dim=0)

        # count the minimum points number in batch
        min_num_points = 100000
        for i in range(batch_size):
            # get num of points in each tensor of batch.
            num_points = np.array(partial_batch[i]).shape[0] # num of points
            if min_num_points > num_points:
                min_num_points = num_points

        # make the number of points in batch the same.
        partial_batch = list(partial_batch) # [batch_size, num_points, channel(x, y, z)]
        for i in range(batch_size):
            num_points_index = np.array(partial_batch[i]).shape[0] # num of points
            # print(f"num = {num_points_index}")
            num_points_index = np.arange(num_points_index)
            # print(f"array ={num_points_index}")
            num_points_index = np.random.permutation(num_points_index) 
            print(f"random = {num_points_index}")
            partial_batch[i] = partial_batch[i][num_points_index[0:min_num_points],:]
        partial_batch = torch.stack(partial_batch, dim=0)

        return comp_batch, partial_batch

if __name__ == "__main__":
    pc_dataset = MakeDataset("./data/ShapeNetCompletion", "chair", "train", 8)
    # i = 46000
    i = 1
    print(len(pc_dataset))
    print(pc_dataset[i][0].size())
    print(pc_dataset[i][1].size())
    min = 10000
    for i in range(4600):
        num_point = pc_dataset[i][1].size()[0]
        if min > num_point:
            min = num_point
    print(min)

    # o3d.visualization.draw_geometries([pc_dataset[7][3]])