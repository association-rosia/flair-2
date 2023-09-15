import argparse
import os
import shutil
import sys
from time import time

import wandb

sys.path.append(os.curdir)

import torch

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.make_dataset import get_list_images

from src.models.lightning import FLAIR2Lightning
from pytorch_lightning import Trainer, callbacks, loggers

from src.constants import get_constants

cst = get_constants()

torch.set_float32_matmul_precision('high')


def main():
    """
    Main function to train the FLAIR-2 model using PyTorch Lightning and WandB.
    """
    # Initialize WandB logging
    init_wandb()

    # Set torch seed
    torch.manual_seed(wandb.config.seed)

    # Load labels and image lists
    df = pd.read_csv(os.path.join(cst.path_data, 'labels-statistics-12.csv'))
    list_images_train = get_list_images(cst.path_data_train)

    list_images_train, list_images_val = train_test_split(list_images_train,
                                                          test_size=0.01,
                                                          random_state=wandb.config.seed)

    list_images_test = get_list_images(cst.path_data_test)

    # Initialize FLAIR-2 Lightning model
    lightning_model = FLAIR2Lightning(
        arch_lib=wandb.config.arch_lib,
        arch=wandb.config.arch,
        encoder_name=wandb.config.encoder_name,
        classes=df['Class'],
        learning_rate=wandb.config.learning_rate,
        class_weights=wandb.config.class_weights,
        list_images_train=list_images_train,
        list_images_val=list_images_val,
        list_images_test=list_images_test,
        aerial_list_bands=wandb.config.aerial_list_bands,
        sen_size=wandb.config.sen_size,
        sen_temp_size=wandb.config.sen_temp_size,
        sen_temp_reduc=wandb.config.sen_temp_reduc,
        sen_list_bands=wandb.config.sen_list_bands,
        prob_cover=wandb.config.prob_cover,
        use_augmentation=wandb.config.use_augmentation,
        train_batch_size=wandb.config.train_batch_size,
        test_batch_size=wandb.config.test_batch_size,
    )

    # Initialize the PyTorch Lightning Trainer
    trainer = init_trainer()

    print()
    if wandb.config.use_augmentation:
        if wandb.config.tta_limit is None:
            # Find optimal TTA limit for inference
            lightning_model = find_optimal_tta_limit(lightning_model, trainer)
        else:
            lightning_model.tta_limit = wandb.config.tta_limit

    # Train the model
    trainer.fit(model=lightning_model)

    # Finish the WandB run
    wandb.finish()


def find_optimal_tta_limit(lightning_model, trainer):
    optimal_tta_limit_found = False

    while not optimal_tta_limit_found:
        # create path to save images
        path_test = os.path.join(cst.path_submissions, 'test')
        # update path in the PL model
        lightning_model.path_predictions = path_test
        # create the folder
        os.makedirs(path_test, exist_ok=True)

        # start the testing
        print(f'Tested TTA limit = {lightning_model.tta_limit}')
        start = time()
        trainer.test(model=lightning_model)
        end = time()

        # remove the saved folder
        shutil.rmtree(path_test)
        inference_time_seconds = end - start - 4
        max_inference_time_seconds = 14 * 60 + 52  # 14 min 52 seconds

        if inference_time_seconds <= max_inference_time_seconds:
            lightning_model.tta_limit += 1
        else:
            optimal_tta_limit_found = True
            lightning_model.tta_limit -= 1

    print(f'Optimal TTA limit =  = {lightning_model.tta_limit}')
    wandb.config['tta_limit'] = lightning_model.tta_limit

    return lightning_model


def init_wandb():
    """
    Initialize WandB logging and configuration.
    """

    # Define the parameters
    parser = argparse.ArgumentParser(description='Script Description')

    # Add the parameters with default values
    parser.add_argument('--arch_lib', type=str, default='', help='Name of the segmentation librairy used')
    parser.add_argument('--arch', type=str, default='', help='Name of the segmentation architecture')
    parser.add_argument('--encoder_name', type=str, default='', help='Name of the timm encoder')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Value of Learning rate')
    parser.add_argument('--min_delta', type=float, default=0.01, help='Value of early stopping minimum delta')
    parser.add_argument('--patience', type=int, default=3, help='Value of early stopping patience')
    parser.add_argument('--aerial_list_bands', nargs='+', type=str,
                        default=['R', 'G', 'B', 'NIR', 'DSM'],
                        help='List of sentinel bands to use')
    parser.add_argument('--sen_size', type=int, default=40, help='Size of the Sentinel 2 images')
    parser.add_argument('--sen_temp_size', type=int, default=3, help='Size of temporal channel for Sentinel 2 images')
    parser.add_argument('--sen_temp_reduc', type=str, default='median', choices=['median', 'mean'],
                        help='Temporal sentinel reduction method (median or mean)')
    parser.add_argument('--sen_list_bands', nargs='+', type=str,
                        default=['2', '3', '4', '5', '6', '7', '8', '8a', '11', '12'],
                        help='List of sentinel bands to use')
    parser.add_argument('--prob_cover', type=int, default=10,
                        help='Probability value that the pixel is covered by cloud or snow.')
    parser.add_argument('--train_batch_size', type=int, default=16, help='Size of each train mini-batch')
    parser.add_argument('--test_batch_size', type=int, default=10, help='Size of each test mini-batch')
    parser.add_argument('--use_augmentation', action='store_true', default=False, help='Use data augmentation & tta')
    parser.add_argument('--class_weights', nargs='+', type=float,
                        default=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                        help='Class weights applied to the cross-entropy loss')
    parser.add_argument('--tta-limit', type=int, default=None, help='TTA limit used')
    parser.add_argument('--seed', type=int, default=42, help='Seed for random initialization')
    parser.add_argument('--max_epochs', type=int, default=30, help='Maximum number of epochs for training')
    parser.add_argument('--dry', action='store_true', default=False, help='Enable or disable dry mode pipeline')

    # Parse the arguments
    args = parser.parse_args()

    # Initialize WandB with project and entity information
    wandb.init(
        entity='association-rosia',
        project='flair-2',
        config=args
    )


def init_trainer() -> Trainer:
    """
    Initialize the PyTorch Lightning Trainer with appropriate configurations.

    Returns:
        trainer (Trainer): Initialized PyTorch Lightning Trainer.
    """
    # Create checkpoint's directory if it doesn't exist
    os.makedirs(cst.path_models, exist_ok=True)

    # Initialize ModelCheckpoint callback to save the best model checkpoint
    checkpoint_callback = callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor='val/miou',
        mode='max',
        dirpath=cst.path_models,
        filename=f'{wandb.run.name}-{wandb.run.id}',
        auto_insert_metric_name=False,
        verbose=True
    )

    early_stopping_callback = callbacks.EarlyStopping(
        monitor='val/miou',
        min_delta=wandb.config.min_delta,
        patience=wandb.config.patience,
        verbose=True,
        mode='max'
    )

    if wandb.config.dry:
        # Configure Trainer for dry run
        trainer = Trainer(
            max_epochs=1,
            logger=loggers.WandbLogger(),
            callbacks=[checkpoint_callback],
            accelerator=cst.device,
            limit_train_batches=1,
            limit_val_batches=1,
        )

    else:
        # Configure Trainer for regular training
        trainer = Trainer(
            max_epochs=wandb.config.max_epochs,
            logger=loggers.WandbLogger(),
            callbacks=[checkpoint_callback, early_stopping_callback],
            accelerator=cst.device
        )

    return trainer


if __name__ == '__main__':
    main()