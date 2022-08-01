import torch
import torch.nn as n
from torch.utils.data import DataLoader
from pytorch3d.loss import chamfer_distance
import argparse
from tqdm import tqdm
from model import *
from data import *
import os

# ----------------------------------------------------------------------------------------
# get options
parser = argparse.ArgumentParser(description="Point Completion Network")
parser.add_argument("--num_points", default=2048)
parser.add_argument("--emb_dim", default=1024)
parser.add_argument("--num_coarse", default=1024)
parser.add_argument("--grid_size", default=4)
parser.add_argument("-b", "--batch_size", default=34)
parser.add_argument("--epochs", default=200)
parser.add_argument("--optimaizer", default="Adam")
parser.add_argument("--dataset_path", default="./data/ShapeNetCompletion")
parser.add_argument("--subset", default="chair")
parser.add_argument("--device", default="cuda")
args = parser.parse_args()
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# make collate function for dataloader
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
        num_points_index = np.arange(num_points_index)
        num_points_index = np.random.permutation(num_points_index) 
        partial_batch[i] = partial_batch[i][num_points_index[0:min_num_points],:]
    partial_batch = torch.stack(partial_batch, dim=0)

    return comp_batch, partial_batch

# load data 
# dataset_path = os.path.join(os.getcwd(), args.dataset_path)
dataset_path = os.path.join(args.dataset_path)
train_dataset = MakeDataset(
    dataset_path=dataset_path,
    subset=args.subset,
    eval="train",
    num_partial_pattern=0
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    # num_workers=3,
    shuffle=True,
    drop_last=True,
    collate_fn=original_collate
) # DataLoader is iterable object.
for i, points in enumerate(tqdm(train_dataloader)):
    print(f"complete points:{points[0].shape},  partial points:{points[1].shape}")
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# prepare subroutine for training one epoch
def train_one_epoch(device, model, train_dataloader, optimazer):
    for i, points in enumerate(tqdm(train_dataloader)):
        a = 0
        
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# main loop
model = PCN(args.num_points, args.num_carse, args.grid_size)

for epoch in range(0, args.epochs):
    model 
