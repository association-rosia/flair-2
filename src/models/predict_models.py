import os
import sys

sys.path.append(os.curdir)

from src.models.lightning import FLAIR2Lightning

import argparse

from src.constants import get_constants

cst = get_constants()


def create_list_objects(names):
    models = []
    dataloaders = []

    for name in names:
        lightning_ckpt = os.path.join(cst.path_models, f'{name}.ckpt')
        lightning_model = FLAIR2Lightning.load_from_checkpoint(lightning_ckpt)
        models.append(lightning_model.model.cuda())
        dataloaders.append(lightning_model.test_dataloader())

    iterators = [iter(loader) for loader in dataloaders]

    return models, iterators


def predict(models, iterators):
    for batches in zip(*iterators):
        for i, batch in enumerate(batches):
            image_ids, aerial, sen, _ = batch
            aerial = aerial.cuda()
            sen = sen.cuda()

            outputs = models[i](aerial=aerial, sen=sen)
            outputs = outputs.softmax(dim=1)
            outputs = outputs.argmax(dim=1)

            print(outputs.shape)

        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for creating submissions with specified models names')
    parser.add_argument('-n', '--names', nargs='+', type=str, help='Name of models to use for submissions')
    args = parser.parse_args()

    models, iterators = create_list_objects(args.names)
    predict(models, iterators)
