import os
import sys

sys.path.append(os.curdir)

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.data.make_dataset import FLAIR2Dataset, get_list_images

from src.models.lightning import FLAIR2Lightning

import argparse
from tqdm import tqdm
import tifffile as tiff
from math import floor
import shutil

from src.constants import get_constants

cst = get_constants()

torch.set_float32_matmul_precision('medium')


def create_list_objects(names, weights, test_batch_size, test_num_workers):
    models = []

    if not weights:
        weights = [1 for _ in names]

    for name in names:
        lightning_ckpt = os.path.join(cst.path_models, f'{name}.ckpt')
        lightning_model = FLAIR2Lightning.load_from_checkpoint(lightning_ckpt)
        lightning_model.test_batch_size = test_batch_size
        lightning_model.test_num_workers = test_num_workers

        # load model
        model = lightning_model.model.half().cuda()
        models.append(model)

    # load dataloader
    # dataloader = lightning_model.test_dataloader()
    list_images_test = get_list_images(cst.path_data_test)

    dataset_test = FLAIR2Dataset(
        list_images=list_images_test,
        aerial_list_bands=['R', 'G', 'B'],
        sen_size=40,
        sen_temp_size=6,
        sen_temp_reduc='median',
        sen_list_bands=['2', '3', '4', '5', '6', '7', '8', '8a', '11', '12'],
        prob_cover=10,
        use_augmentation=True,
        use_tta=False,
        is_val=False,
        is_test=True,
    )

    dataloader = DataLoader(
        dataset=dataset_test,
        batch_size=test_batch_size,
        num_workers=test_num_workers,
        shuffle=False,
        drop_last=False,
    )

    return models, weights, dataloader


def predict(models, weights, dataloader, path_predictions, save_predictions):
    print(f'\nInference - save_predictions = {save_predictions}')

    for batch in tqdm(dataloader, total=len(dataloader)):
        outputs = torch.zeros((len(batch[0]), 13, 512, 512)).cuda()
        image_ids, aerial, sen, _ = batch
        aerial = aerial.half().cuda()
        sen = sen.half().cuda()

        for i, model in enumerate(models):
            output = model(aerial=aerial, sen=sen)
            output = output.softmax(dim=1)
            output = torch.mul(float(weights[i]), output)
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
    parser.add_argument('-w', '--weights', nargs='+', type=str, help='Weights of models to use for submissions')
    args = parser.parse_args()

    run_names = [name.split('-')[0] for name in args.names]  # Bugfix: OSError: [Errno 36] File name too long
    run_names = '_'.join(run_names)
    path_predictions = os.path.join(cst.path_submissions, run_names)
    os.makedirs(path_predictions, exist_ok=True)

    test_batch_size = 10
    test_num_workers = 10

    models, weights, dataloader = create_list_objects(args.names,
                                                      args.weights,
                                                      test_batch_size,
                                                      test_num_workers)

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    start.record()
    predict(models, weights, dataloader, path_predictions, save_predictions=False)
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()
    inference_time_seconds = start.elapsed_time(end) / 1000.0
    minutes = floor(inference_time_seconds // 60)
    seconds = floor(inference_time_seconds % 60)
    submission_inference_time = f'{minutes}-{seconds}'

    print(f'>>>>>>> inference_time_seconds = {inference_time_seconds}')

    # predict(models, weights, dataloader, path_predictions, save_predictions=True)

    name_submission = f'{run_names}_{cst.baseline_inference_time}_{submission_inference_time}'
    zip_path_submission = os.path.join(cst.path_submissions, name_submission)
    shutil.make_archive(zip_path_submission, 'zip', path_predictions)
    shutil.rmtree(path_predictions)
