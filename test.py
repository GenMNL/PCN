import numpy as np
import torch
import open3d as o3d
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
from data import *

def original_collate(batch_list):
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

path = "./data/ShapeNetCompletion/"
train_dataset = MakeDataset(
    dataset_path=path,
    subset="chair",
    eval="train",
    num_partial_pattern=8
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    collate_fn=original_collate
)

count = 0
for comp, partial in train_dataloader:
    print(np.array(comp).shape)
    print(np.array(partial).shape)
    # print(torch.tensor(partial))
    count += 1
    if count >= 4:
        break