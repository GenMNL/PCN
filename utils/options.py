import argparse

# ----------------------------------------------------------------------------------------
def make_parser():
    parser = argparse.ArgumentParser(description="options of PCN")

    # make parser for train (part of this is used for test)
    parser.add_argument("--emb_dim", default=1024, type=int)
    parser.add_argument("--num_coarse", default=1024, type=int)
    parser.add_argument("--num_comp", default=16384, type=int)
    parser.add_argument("--grid_size", default=4, type=int)
    parser.add_argument("--batch_size", default=3, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--lr", default=1e-3, help="learning rate", type=float)
    parser.add_argument("--dataset_dir", default="./../../dataset/BridgeCompletionDataset-v2/all")
    parser.add_argument("--save_dir", default="./checkpoint")
    parser.add_argument("--subset", default="bridge")
    parser.add_argument("--device", default="cuda")

    # make parser for test
    parser.add_argument("--result_dir", default="./result")
    parser.add_argument("--select_result", default="best") # you can select best or normal
    parser.add_argument("--result_subset", default="bridge")
    parser.add_argument("--result_eval", default="test")
    parser.add_argument("--year", default="2023")
    parser.add_argument("-d", "--date", type=str)

    # 
    parser.add_argument("--other", default="", type=str)
    parser.add_argument("--result", default="", type=str)
    parser.add_argument("--next", default="", type=str)
    return parser
# ----------------------------------------------------------------------------------------
