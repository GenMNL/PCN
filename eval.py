import torch
from torch.utils.data import DataLoader
import open3d as o3d
import numpy as np
import pandas as pd
import os
from options import *
from model import *
from data import *

# ----------------------------------------------------------------------------------------
# make function
# ----------------------------------------------------------------------------------------
# for export ply
def export_ply(dir_path, file_name, type, point_cloud):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(point_cloud)
    path = os.path.join(dir_path, type, str(file_name)+".ply")
    o3d.io.write_point_cloud(path, pc)

def resize(ary, max, min):
    max = max[0,:]
    min = min[0,:]
    ary[:,0] *= (max[0] - min[0])
    ary[:,1] *= (max[1] - min[1])
    ary[:,2] *= (max[2] - min[2])
    ary[:,0] += min[0]
    ary[:,1] += min[1]
    ary[:,2] += min[2]

    return ary

# for test
def test(device, model, dataloader, len_dataset, save_dir):
    model.eval()
    total_loss = 0.0
    count =0

    feature_df = pd.DataFrame(np.zeros((len_dataset, args.emb_dim)), index=np.arange(1, len_dataset+1),
                              columns=np.arange(1, args.emb_dim+1))
    with torch.no_grad():
        for i, points in enumerate(dataloader):
            comp, partial = points[0], points[1]
            comp_max, comp_min, partial_max, partial_min = points[2], points[3], points[4], points[5]
            # prediction
            feature_v, coarse, fine = model(partial)

            feature_v = feature_v.detach().cpu().numpy()
            feature_v = feature_v.reshape(args.emb_dim)
            feature_df.loc[i+1,:] = feature_v

            comp = comp.detach().cpu().numpy()
            comp = comp.reshape(args.num_coarse*(args.grid_size**2), -1)
            comp = resize(comp, comp_max, comp_min)
            partial = partial.detach().cpu().numpy()
            partial = partial.reshape(args.num_points, -1)
            partial = resize(partial, partial_max, partial_min)
            fine = fine.detach().cpu().numpy()
            fine = fine.reshape(args.num_coarse*(args.grid_size**2), -1)
            fine = resize(fine, comp_max, comp_min)
            coarse = coarse.detach().cpu().numpy()
            coarse = coarse.reshape(args.num_coarse, -1)
            coarse = resize(coarse, comp_max, comp_min)
            export_ply(save_dir, i+1, "comp", comp) # save point cloud of comp
            export_ply(save_dir, i+1, "partial", partial) # save point cloud of partial
            export_ply(save_dir, i+1, "fine", fine) # save point cloud of fine
            export_ply(save_dir, i+1, "coarse", coarse) # save point cloud of coarse

    feature_path = os.path.join(args.result_dir, args.result_subset, "emb.csv")
    feature_df.to_csv(feature_path)
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get options
    parser = make_parser()
    args = parser.parse_args()

    # make result dirctory
    result_dir = os.path.join(args.result_dir, args.subset)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # make test dataset
    data_dir = os.path.join(args.dataset_dir)
    test_dataset = MakeDataset(dataset_path=data_dir, subset=args.subset,
                               eval=args.result_eval, num_partial_pattern=0, device=args.device)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, # the batch size of test must be 1
                                 collate_fn=OriginalCollate(args.num_points, args.num_comp, args.device))
    len_dataset = len(test_dataset)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # load model
    # you can't change here because this is same with train
    model = PCN(args.num_points, args.emb_dim, args.num_coarse, args.grid_size, args.device).to(args.device)
    pth_path = os.path.join(args.save_dir, args.result_subset, args.year, args.date, args.select_result + "_weight.tar")

    checkpoint = torch.load(pth_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    result_dir = os.path.join(args.result_dir, args.result_subset)
    test(args.device, model, test_dataloader, len_dataset, result_dir)
