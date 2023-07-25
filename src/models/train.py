import wandb

import os
import sys

sys.path.append('.')

import pandas as pd

from torch.utils.data import DataLoader, Subset
from src.data.make_dataset import FLAIR2Dataset, get_list_images

from src.models.lightning import FLAIR2Lightning
import pytorch_lightning as pl

import src.constants as cst


def main():
    wandb.init(
        entity='urgellbapt',
        project='developpement',
        group='FLAIR2',
        config={
            'architecture': 'Unet',
            'encoder_name': 'tu-efficientnetv2_xl',
            'encoder_weight': None,
            'learning_rate': 1e-4,
        }
    )
    
    df = pd.read_csv(os.path.join(cst.PATH_DATA, 'labels-statistics.csv'))
    
    path_train = cst.PATH_DATA_TRAIN
    list_images_train = get_list_images(path_train)
    dataset = FLAIR2Dataset(
        list_images=list_images_train,
        sen_size=40,
        is_test=False,
        use_augmentation=False,
    )
    
    # TODO: Implement a smarter spliting strategy
    split_index = int(len(dataset) * 0.9)
    dataloader_train = DataLoader(
        dataset=Subset(dataset, range(split_index)),
        batch_size=16, 
        shuffle=True, 
        drop_last=True
    )
    
    dataloader_val = DataLoader(
        dataset=Subset(dataset, range(split_index, len(dataset))),
        batch_size=16, 
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
    
    os.makedirs(cst.PATH_MODELS, exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor='val/loss',
        mode='min',
        dirpath=cst.PATH_MODELS,
        filename=f'{wandb.config.architecture}-{wandb.config.encoder_name}-{wandb.run.name}-{wandb.run.id}',
        auto_insert_metric_name=False,
        verbose=True
    )

    n_epochs = 3#0
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
    
if __name__ == '__main__':
    main()