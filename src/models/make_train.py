import wandb

import os
import sys

sys.path.append('.')

import pandas as pd
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
from src.data.make_dataset import FLAIR2Dataset, get_list_images

from src.models.lightning import FLAIR2Lightning
import pytorch_lightning as pl

from src.constants import get_constants

cst = get_constants()


def main():

    wandb.init(
        entity='association-rosia',
        project='FLAIR-2',
        config={
            'architecture': 'Unet',
            'encoder_name': 'tu-efficientnetv2_xl',  # TODO: try different size
            'encoder_weight': None,
            'learning_rate': 1e-4,
            'sen_size': 40,
            'use_augmentation': False,
            'batch_size': 16
        }
    )

    df = pd.read_csv(os.path.join(cst.path_data, 'labels-statistics.csv'))
    # TODO: Implement a smarter splitting strategy
    list_images_train = get_list_images(cst.path_data_train)
    list_images_train, list_images_val = train_test_split(list_images_train, test_size=0.1, random_state=42)

    dataset_train = FLAIR2Dataset(
        list_images=list_images_train,
        sen_size=wandb.config.sen_size,
        is_test=False,
        use_augmentation=wandb.config.use_augmentation,
    )

    dataset_val = FLAIR2Dataset(
        list_images=list_images_val,
        sen_size=wandb.config.sen_size,
        is_test=False,
        use_augmentation=wandb.config.use_augmentation,
    )

    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=wandb.config.batch_size,
        shuffle=True,
        drop_last=True
    )

    dataloader_val = DataLoader(
        dataset=dataset_val,
        batch_size=wandb.config.batch_size,
        shuffle=False,
        drop_last=True
    )

    lightning_model = FLAIR2Lightning(
        architecture=wandb.config.architecture,
        encoder_name=wandb.config.encoder_name,
        encoder_weight=wandb.config.encoder_weight,
        classes=df['Class'],
        learning_rate=wandb.config.learning_rate,
        criterion_weight=df['Freq.-train (%)']
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

    n_epochs = 3  # 0
    trainer = pl.Trainer(
        max_epochs=n_epochs,
        logger=pl.loggers.WandbLogger(),
        callbacks=[checkpoint_callback],
        # accelerator='gpu',
        accelerator='cpu',
        limit_train_batches=3,
        limit_val_batches=3,
        limit_test_batches=3,
    )

    trainer.fit(
        model=lightning_model,
        train_dataloaders=dataloader_train,
        val_dataloaders=dataloader_val,
    )
    
    list_images_test = get_list_images(cst.path_data_test)
    
    dataset_test = FLAIR2Dataset(
        list_images=list_images_test,
        sen_size=wandb.config.sen_size,
        is_test=True,
        use_augmentation=False,
    )
    
    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=wandb.config.batch_size,
        shuffle=False,
        drop_last=False
    )
    
    
    lightning_ckpt = f'{wandb.run.name}-{wandb.run.id}.ckpt'
    lightning_model = FLAIR2Lightning.load_from_checkpoint(lightning_ckpt)
    trainer.predict(
        model=lightning_model,
        dataloaders=dataloader_test
    )
    
    lightning_ckpt = f'{wandb.run.name}-{wandb.run.id}.ckpt'
    
    lightning_model = FLAIR2Lightning.load_from_checkpoint(lightning_ckpt)
    path_submissions = os.path.join(cst.path_submissions, f'{wandb.run.name}-{wandb.run.id}')
    lightning_model.path_submissions = path_submissions
    os.makedirs(path_submissions, exist_ok=False)
    trainer.predict(
        model=lightning_model,
        dataloaders=dataloader_test
    )
    
    lightning_model = FLAIR2Lightning.load_from_checkpoint(lightning_ckpt)
    path_submissions = os.path.join(cst.path_submissions, f'tta-{wandb.run.name}-{wandb.run.id}')
    lightning_model.path_submissions = path_submissions
    lightning_model.apply_tta = True
    os.makedirs(path_submissions, exist_ok=False)
    trainer.predict(
        model=lightning_model,
        dataloaders=dataloader_test
    )
    
    wandb.finish()

if __name__ == '__main__':
    main()
