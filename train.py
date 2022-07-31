import torch
import torch.nn as n
from torch.utils.data import DataLoader
import argparse
from model import *
from data import *
import os

# ----------------------------------------------------------------------------------------
# get options
parser = argparse.ArgumentParser(description="Point Completion Network")
parser.add_argument("--num_points", default=1024)
parser.add_argument("--emb_dim", default=1024)
parser.add_argument("--num_coarse", default=1024)
parser.add_argument("--grid_size", default=4)
parser.add_argument("-b", "--batch_size", default=34)
parser.add_argument("--epochs", default=200)
parser.add_argument("--dataset_path", default="./data")
parser.add_argument("--subset", default="chair")
parser.add_argument("--device", default="cuda")
args = parser.parse_args()
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
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
    shuffle=True
    # drop_last=True
) # DataLoader is iterable object.

lc = 0
lp = 0
for i, (comp, partial) in enumerate(train_dataloader):
    lc += len(comp)
    lp += len(partial)
print(f"the length of comp data is :{lc}")
print(f"the length of partial data is :{lp}")

# model = PCN(args.num_points, args.num_carse, args.grid_size)
