import os
import sys

import wandb

sys.path.append(os.curdir)

import torch

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.make_dataset import get_list_images

from src.models.lightning import FLAIR2Lightning
import pytorch_lightning as pl

from src.constants import get_constants

cst = get_constants()

torch.set_float32_matmul_precision('medium')  # try 'high'


def main():
    wandb.init(
        entity='association-rosia',
        project='flair-2',
        config={
            'architecture': 'Unet',
            'encoder_name': 'tu-efficientnetv2_m',
            'encoder_weight': None,
            'learning_rate': 1e-3,
            'sen_size': 40,
            'batch_size': 16,
            'use_augmentation': False
        }
    )

    df = pd.read_csv(os.path.join(cst.path_data, 'labels-statistics-12.csv'))
    # TODO: Implement a smarter splitting strategy
    list_images_train = get_list_images(cst.path_data_train)
    list_images_train, list_images_val = train_test_split(list_images_train, test_size=0.1, random_state=42)
    list_images_test = get_list_images(cst.path_data_test)

    lightning_model = FLAIR2Lightning(
        architecture=wandb.config.architecture,
        encoder_name=wandb.config.encoder_name,
        encoder_weight=wandb.config.encoder_weight,
        classes=df['Class'],
        learning_rate=wandb.config.learning_rate,
        criterion_weight=df['Freq.-train (%)'],
        list_images_train=list_images_train,
        list_images_val=list_images_val,
        list_images_test=list_images_test,
        sen_size=wandb.config.sen_size,
        use_augmentation=wandb.config.use_augmentation,
        batch_size=wandb.config.batch_size,
    )

    os.makedirs(cst.path_models, exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor='val/loss',
        mode='min',
        dirpath=cst.path_models,
        filename=f'{wandb.run.name}-{wandb.run.id}',
        auto_insert_metric_name=False,
        verbose=True
    )

    n_epochs = 10
    trainer = pl.Trainer(
        max_epochs=n_epochs,
        logger=pl.loggers.WandbLogger(),
        callbacks=[checkpoint_callback],
        accelerator='gpu',
        # fast_dev_run=3,
        # limit_train_batches=3,
        # limit_val_batches=3,
    )

    trainer.fit(model=lightning_model)
    wandb.finish()


if __name__ == '__main__':
    main()
