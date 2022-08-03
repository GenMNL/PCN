import torch
import torch.nn as n
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
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
parser.add_argument("--optimaizer", default="Adam", help="if you want to choose other optimization, you must change the code.")
parser.add_argument("--lr", default=1e-4, help="learning rate")
parser.add_argument("--dataset_dir", default="./data/ShapeNetCompletion")
parser.add_argument("--save_path", default="./checkpoint")
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

class OriginalCollate():
    def __init__(self, num_points):
        self.num_points = num_points
        # os.environ["TOKENIZERS_PARALLELISM"] = "true"

    def __call__(self, batch_list):
        # get batch size
        # batch_size = np.array(batch_list).shape[0]
        batch_size = len(batch_list)

        # transform tuple of complete point cloud to tensor
        comp_batch, partial_batch = list(zip(*batch_list))
        comp_batch = torch.stack(comp_batch, dim=0).to(args.device)

        # transform tuple of partial point cloud to tensor
        partial_batch = list(partial_batch)
        for i in range(batch_size):
            n = len(partial_batch[i])
            idx = np.random.permutation(n)
            if len(idx) < self.num_points:
                temp = np.random.randint(0, n, size=(self.num_points - n))
                idx = np.concatenate([idx, temp])
            partial_batch[i] = partial_batch[i][idx[:self.num_points], :]

        partial_batch = torch.stack(partial_batch, dim=0).to(args.device)

        return comp_batch, partial_batch


# load data 
# train data
data_dir = os.path.join(args.dataset_dir)
train_dataset = MakeDataset(
    dataset_path=data_dir,
    subset=args.subset,
    eval="train",
    num_partial_pattern=0,
    device=args.device
)
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    # num_workers=3,
    shuffle=True,
    drop_last=True,
    collate_fn=OriginalCollate(args.num_points)
) # DataLoader is iterable object.

# validation data
val_dataset = MakeDataset(
    dataset_path=data_dir,
    subset=args.subset,
    eval="val",
    num_partial_pattern=0,
    device=args.device
)
val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    collate_fn=OriginalCollate(args.num_points)
)

# check of data in dataloader
# for i, points in enumerate(tqdm(train_dataloader)):
    # print(f"complete points:{points[0].shape},  partial points:{points[1].shape}")
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# prepare subroutine for training one epoch
def train_one_epoch(device, model, dataloader, alpha, optim):
    model.train()
    train_loss = 0.0
    count = 0

    for i, points in enumerate(tqdm(dataloader)):
        comp = points[0]
        partial = points[1]
        # prediction
        feature_v, coarse, fine = model(partial)
        # get chamfer distance loss
        CD_coarse = chamfer_distance(coarse, comp)
        CD_fine = chamfer_distance(fine, comp)
        CD_loss = CD_coarse[0] + alpha*CD_fine[0]

        # backward + optim
        optim.zero_grad()
        CD_loss.backward()
        optim.step()

        train_loss += CD_loss
        count += 1

    train_loss = float(train_loss)/count
    return train_loss

def val_one_epoch(device, model, dataloader):
    model.eval()
    val_loss = np.inf
    count = 0

    for i, points in enumerate(dataloader):
        comp = points[0]
        partial = points[1]
        # prediction
        feature_v, coarse, fine = model(partial)
        # get chamfer distance loss
        CD_loss = chamfer_distance(fine, comp)

        val_loss += CD_loss[0]
        count += 1

    val_loss = float(val_loss)/count
    return val_loss

# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# main loop
model = PCN(args.num_points, args.emb_dim,args.num_coarse, args.grid_size, args.device).to(args.device)
if args.optimaizer == "Adam":
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=[0.9, 0.999])
lr_schdual = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.7)
writter = SummaryWriter()

best_loss = np.inf
for epoch in tqdm(range(1, args.epochs+1)):

    # determin the ration of loss
    if epoch < 50:
        alpha = 0.01
    elif epoch < 100:
        alpha = 0.1
    elif epoch < 150:
        alpha = 0.5
    else:
        alpha = 1.0
    # get loss of one epoch
    train_loss = train_one_epoch(args.device, model, train_dataloader, alpha, optim)
    val_loss = val_one_epoch(args.device, model, val_dataloader)
    writter.add_scalar("train_loss", train_loss, epoch)
    writter.add_scalar("validation_loss", val_loss, epoch)

    # if val loss is better than best loss, update best loss to val loss
    if val_loss < best_loss:
        best_loss = val_loss
        bl_path = os.path.join(args.save_dir, "best_weight.pth")
        torch.save({
                    'epoch':epoch,
                    'model_state_dict':model.state_dict(), 
                    'optimizer_state_dict':optim.state_dict(),
                    'loss':best_loss
                    }, bl_path)
    # save normal weight 
    nl_path = os.path.join(args.save_dir, "normal_weight.pth")
    torch.save({
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optim.state_dict(),
                'loss':val_loss
                }, nl_path)
    lr_schdual.step()

# close writter
writter.close()