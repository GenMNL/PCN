import torch
from torch.utils.data import dataset
from torch.utils.data import dataloader
import argparse



if __name__ == "__main__":

    # get options
    parser = argparse.ArgumentParser(description="Test of PCN")
    parser.add_argument("")


    class OriginalCollate():
        def __init__(self):
            pass

        def __call__(self, *args: Any, **kwds: Any) -> Any:
            pass

    dataset =