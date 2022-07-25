import torch
import argparse
import torch.nn as n
from model import *

# ----------------------------------------------------------------------------------------
# get options
parser = argparse.ArgumentParser(description="Point Completion Network")
parser.add_argument("--num_points", default=1024)
parser.add_argument("--emb_dim", default=1024)
parser.add_argument("--num_coarse", default=1024)
parser.add_argument("--grid_size", default=4)
parser.add_argument("-b", "--batchsize", default=34)
parser.add_argument("--epochs", default=200)
parser.add_argument("--dataset_path", default="./data/")
parser.add_argument("--device", default="cpu")
args = parser.parse_args()
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# load data
model = PCN(args.num_points, args.num_carse, args.grid_size)
