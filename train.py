import torch
import torch.nn as n
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from pytorch3d.loss import chamfer_distance
import argparse
from tqdm import tqdm
from model import *
from data import *
from options import *
import os

# ----------------------------------------------------------------------------------------
# prepare subroutine for training one epoch
def train_one_epoch(device, model, dataloader, alpha, optim):
    model.train()
    train_loss = 0.0
    count = 0

    for i, points in enumerate(tqdm(dataloader, desc="train")):
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
    val_loss = 0.0
    count = 0

    with torch.no_grad():
        for i, points in enumerate(tqdm(dataloader, desc="validation")):
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
if __name__ == "__main__":
    #  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # get options
    parser = make_parser()
    args = parser.parse_args()
    #  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    #  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # make dataloader
    # data_dir = os.path.join(args.dataset_dir)
    train_dataset = MakeDataset(
        dataset_path=args.dataset_dir,
        subset=args.subset,
        eval="train",
        num_partial_pattern=4,
        device=args.device
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=OriginalCollate(args.num_points, args.num_comp, args.device)
    ) # DataLoader is iterable object.

    # validation data
    val_dataset = MakeDataset(
        dataset_path=args.dataset_dir,
        subset=args.subset,
        eval="val",
        num_partial_pattern=4,
        device=args.device
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        collate_fn=OriginalCollate(args.num_points, args.num_comp, args.device)
    )

    # check of data in dataloader
    # for i, points in enumerate(tqdm(train_dataloader)):
        # print(f"complete points:{points[0].shape},  partial points:{points[1].shape}")
    #  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    #  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # prepare model and optimaizer
    model = PCN(args.num_points, args.emb_dim,args.num_coarse, args.grid_size, args.device).to(args.device)
    if args.optimaizer == "Adam":
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=[0.9, 0.999])
    lr_schdual = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.7)
    writter = SummaryWriter()
    #  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    #  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # main loop
    best_loss = np.inf
    for epoch in tqdm(range(1, args.epochs+1), desc="main loop"):

        # determin the ration of loss
        if epoch < 50:
            alpha = 0.01
        elif epoch < 100:
            alpha = 0.1
        elif epoch < 200:
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
            bl_path = os.path.join(args.save_dir, args.subset, "best_weight.tar")
            torch.save({
                        'epoch':epoch,
                        'model_state_dict':model.state_dict(), 
                        'optimizer_state_dict':optim.state_dict(),
                        'loss':best_loss
                        }, bl_path)
        # save normal weight 
        nl_path = os.path.join(args.save_dir, args.subset, "normal_weight.tar")
        torch.save({
                    'epoch':epoch,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optim.state_dict(),
                    'loss':val_loss
                    }, nl_path)
        lr_schdual.step()

    # close writter
    writter.close()
