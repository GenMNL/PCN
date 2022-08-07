import argparse

# ----------------------------------------------------------------------------------------
def parser():
    parser = argparse.ArgumentParser(description="options of PCN")

    # train
    parser.add_argument("--num_points", default=2048)
    parser.add_argument("--emb_dim", default=1024)
    parser.add_argument("--num_coarse", default=1024)
    parser.add_argument("--grid_size", default=4)
    parser.add_argument("--batch_size", default=34)
    parser.add_argument("--epochs", default=200)
    parser.add_argument("--optimaizer", default="Adam", help="if you want to choose other optimization, you must change the code.")
    parser.add_argument("--lr", default=1e-4, help="learning rate")
    parser.add_argument("--dataset_dir", default="./data/ShapeNetCompletion")
    parser.add_argument("--save_dir", default="./checkpoint")
    parser.add_argument("--subset", default="chair")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # test
    return args
# ----------------------------------------------------------------------------------------