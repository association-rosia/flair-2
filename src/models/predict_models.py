import os
import sys
sys.path.append(os.curdir)

from src.data.make_dataset import FLAIR2Dataset
from torch.utils.data import DataLoader

import argparse

from src.constants import get_constants
cst = get_constants()

parser = argparse.ArgumentParser(description='Script for creating submissions with specified models names')
parser.add_argument('-n', '--names', nargs='+', type=str, help='Name of models to use for submissions')
args = parser.parse_args()

print(args.names)

# TODO: create test dataloader

# TODO: load n models in GPUs

# TODO: infer without saving to calculate the inference time

# TODO: infer with saving to create the submission folder
