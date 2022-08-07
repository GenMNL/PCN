import argparse
from options import *

parser = make_parser()

args = parser.parse_args()
print("test")
print(args.num_points)
