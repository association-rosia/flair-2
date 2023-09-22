import os
import sys

import torch
import numpy as np

sys.path.append(os.curdir)

from src.models.lightning import FLAIR2Lightning

import argparse
from tqdm import tqdm
import tifffile as tiff

from src.constants import get_constants

cst = get_constants()


def create_list_objects(names, test_batch_size):
    models = []
    dataloaders = []

    for name in names:
        lightning_ckpt = os.path.join(cst.path_models, f'{name}.ckpt')
        lightning_model = FLAIR2Lightning.load_from_checkpoint(lightning_ckpt)
        lightning_model.test_batch_size = test_batch_size

        # load model
        model = lightning_model.model.cuda()
        models.append(model)

        # load dataloader
        dataloader = lightning_model.test_dataloader()
        dataloaders.append(dataloader)

    iterators = [iter(loader) for loader in dataloaders]

    return models, iterators


def predict(models, iterators, test_batch_size, path_predictions, save_predictions):
    for batches in tqdm(zip(*iterators), total=len(iterators[0])):
        image_ids = None
        outputs = torch.zeros((test_batch_size, 13, 512, 512)).cuda()

        for i, batch in enumerate(batches):
            image_ids, aerial, sen, _ = batch
            # print(image_ids)
            aerial = aerial.cuda()
            sen = sen.cuda()

            output = models[i](aerial=aerial, sen=sen)
            output = output.softmax(dim=1)
            outputs = torch.add(outputs, output)

        outputs = outputs.argmax(dim=1)

        if save_predictions:
            for pred_label, img_id in zip(outputs, image_ids):
                img = pred_label.numpy(force=True)
                img = img.astype(dtype=np.uint8)
                img_path = os.path.join(path_predictions, f'PRED_{img_id}')
                tiff.imwrite(img_path, img, dtype=np.uint8, compression='LZW')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for creating submissions with specified models names')
    parser.add_argument('-n', '--names', nargs='+', type=str, help='Name of models to use for submissions')
    args = parser.parse_args()

    test_batch_size = 16
    path_predictions = ''

    models, iterators = create_list_objects(args.names, test_batch_size)
    predict(models, iterators, test_batch_size, path_predictions, save_predictions=False)
