import os
import sys

import torch
import numpy as np

sys.path.append(os.curdir)

from src.models.lightning import FLAIR2Lightning

import argparse
from tqdm import tqdm
import tifffile as tiff
from math import floor
import shutil

from src.constants import get_constants

cst = get_constants()


def create_list_objects(names, test_batch_size, test_num_workers):
    models = []
    dataloaders = []

    for name in names:
        lightning_ckpt = os.path.join(cst.path_models, f'{name}.ckpt')
        lightning_model = FLAIR2Lightning.load_from_checkpoint(lightning_ckpt)
        lightning_model.test_batch_size = test_batch_size
        lightning_model.test_num_workers = test_num_workers

        # load model
        model = lightning_model.model.cuda()
        models.append(model)

        # load dataloader
        dataloader = lightning_model.test_dataloader()
        dataloaders.append(dataloader)

    iterators_1 = [iter(loader) for loader in dataloaders]
    iterators_2 = [iter(loader) for loader in dataloaders]

    return models, iterators_1, iterators_2


def predict(models, iterators, test_batch_size, path_predictions, save_predictions):
    print(f'\nInference - save_predictions = {save_predictions}')

    for batches in tqdm(zip(*iterators), total=len(iterators[0])):
        image_ids = None
        outputs = torch.zeros((test_batch_size, 13, 512, 512)).cuda()

        for i, batch in enumerate(batches):
            image_ids, aerial, sen, _ = batch
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

    run_names = '_'.join(args.names)
    path_predictions = os.path.join(cst.path_submissions, run_names)
    os.makedirs(path_predictions, exist_ok=True)

    test_batch_size = 5
    test_num_workers = 18

    models, iterators_1, iterators_2 = create_list_objects(args.names, test_batch_size, test_num_workers)

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    start.record()
    predict(models, iterators_1, test_batch_size, path_predictions, save_predictions=False)
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()
    inference_time_seconds = start.elapsed_time(end) / 1000.0
    minutes = floor(inference_time_seconds // 60)
    seconds = floor(inference_time_seconds % 60)
    submission_inference_time = f'{minutes}-{seconds}'

    predict(models, iterators_2, test_batch_size, path_predictions, save_predictions=True)

    name_submission = f'{run_names}_{cst.baseline_inference_time}_{submission_inference_time}'
    zip_path_submission = os.path.join(cst.path_submissions, name_submission)
    shutil.make_archive(zip_path_submission, 'zip', path_predictions)
    shutil.rmtree(path_predictions)
