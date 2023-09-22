import os
import sys
sys.path.append(os.curdir)

from src.models.lightning import FLAIR2Lightning

import argparse

from src.constants import get_constants
cst = get_constants()

parser = argparse.ArgumentParser(description='Script for creating submissions with specified models names')
parser.add_argument('-n', '--names', nargs='+', type=str, help='Name of models to use for submissions')
args = parser.parse_args()

lightning_models = []
dataloaders = []
for name in args.names:
    lightning_ckpt = os.path.join(cst.path_models, f'{name}.ckpt')
    lightning_model = FLAIR2Lightning.load_from_checkpoint(lightning_ckpt)
    lightning_models.append(lightning_model.model.cuda())
    dataloaders.append(lightning_model.test_dataloader())


iterators = [iter(loader) for loader in dataloaders]

for batches in zip(*iterators):
    print(batches)
    break


# TODO: create test dataloader

# TODO: load n models in GPUs

# TODO: infer without saving to calculate the inference time

# TODO: infer with saving to create the submission folder
