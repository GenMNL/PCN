import torch
import torch.nn as n
from torch.utils.data import DataLoader
import torch.multiprocessing as multiprocessing
from tensorboardX import SummaryWriter
from pytorch3d.loss import chamfer_distance
import datetime
from tqdm import tqdm
from models.model import *
from utils.data import *
from utils.options import *
import os

# ----------------------------------------------------------------------------------------
# prepare subroutine for training one epoch
def train_one_epoch(model, dataloader, alpha, optim):
    model.train()
    train_loss = 0.0

    for i, points in enumerate(tqdm(dataloader, desc="train")):
        comp = points[0]
        partial = points[1]
        # prediction
        _, coarse, fine = model(partial)
        # get chamfer distance loss
        CD_coarse, _ = chamfer_distance(coarse, comp)
        CD_fine, _ = chamfer_distance(fine, comp)
        CD_loss = CD_coarse + alpha*CD_fine

        # backward + optim
        optim.zero_grad()
        CD_loss.backward()
        optim.step()

        train_loss = train_loss + CD_loss

    train_loss = float(train_loss)/len(dataloader)
    return train_loss

def val_one_epoch(model, dataloader):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, points in enumerate(tqdm(dataloader, desc="validation")):
            comp = points[0]
            partial = points[1]
            # prediction
            _, _, fine = model(partial)
            # get chamfer distance loss
            CD_loss, _ = chamfer_distance(fine, comp)

            val_loss = val_loss + CD_loss

    val_loss = float(val_loss)/len(dataloader)
    return val_loss

# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get options
    parser = make_parser()
    args = parser.parse_args()

    # make path of save params
    dt_now = datetime.datetime.now()
    save_date = str(dt_now.month) + str(dt_now.day) + "-" + str(dt_now.hour) + "-" + str(dt_now.minute)
    save_dir = os.path.join(args.save_dir, args.subset, str(dt_now.year), save_date)
    save_normal_path = os.path.join(save_dir, "normal_weight.tar")
    save_best_path = os.path.join(save_dir, "best_weight.tar")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        # make condition file
        with open(os.path.join(save_dir, "conditions.txt"), 'w') as f:
            json.dump(args.__dict__, f, indent=4)

    writter = SummaryWriter()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # make dataloader
    # data_dir = os.path.join(args.dataset_dir)
    train_dataset = MakeDataset(dataset_path=args.dataset_dir, subset=args.subset,
                                eval="train", num_partial_pattern=2, device=args.device)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                  shuffle=True, drop_last=True, num_workers=4,
                                  collate_fn=OriginalCollate(args.device)) # DataLoader is iterable object.

    # validation data
    val_dataset = MakeDataset(dataset_path=args.dataset_dir, subset=args.subset,
                              eval="val", num_partial_pattern=2, device=args.device)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                                shuffle=True, drop_last=True, num_workers=4,
                                collate_fn=OriginalCollate(args.device))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # prepare model and optimaizer
    model = PCN(args.emb_dim,args.num_coarse, args.grid_size, args.device).to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    if multiprocessing.get_start_method() == 'fork':
        multiprocessing.set_start_method('spawn', force=True)
        print("{} setup done".format(multiprocessing.get_start_method()))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        train_loss = train_one_epoch(model, train_dataloader, alpha, optim)
        val_loss = val_one_epoch(model, val_dataloader)

        writter.add_scalar("train_loss", train_loss, epoch)
        writter.add_scalar("validation_loss", val_loss, epoch)

        # if val loss is better than best loss, update best loss to val loss
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({'epoch':epoch,
                        'model_state_dict':model.state_dict(), 
                        'optimizer_state_dict':optim.state_dict(),
                        'loss':best_loss
                        }, save_best_path)
        # save normal weight 
        torch.save({'epoch':epoch,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optim.state_dict(),
                    'loss':val_loss
                    }, save_normal_path)
        #lr_schdual.step()

    # close writter
    writter.close()

