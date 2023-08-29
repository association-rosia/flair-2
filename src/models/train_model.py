import os
import sys

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
torch.autograd.set_detect_anomaly(True)


def main():
    """
    Main function to train the FLAIR-2 model using PyTorch Lightning and WandB.
    """
    # Initialize WandB logging
    init_wandb()
    
    # Load labels and image lists
    df = pd.read_csv(os.path.join(cst.path_data, 'labels-statistics-12.csv'))
    list_images_train = get_list_images(cst.path_data_train)
    list_images_train, list_images_val = train_test_split(list_images_train, test_size=0.1, random_state=wandb.config.seed)
    list_images_test = get_list_images(cst.path_data_test)

    # Initialize FLAIR-2 Lightning model
    lightning_model = FLAIR2Lightning(
        arch=wandb.config.arch,
        encoder_name=wandb.config.encoder_name,
        classes=df['Class'],
        learning_rate=wandb.config.learning_rate,
        class_weights=wandb.config.class_weights,
        list_images_train=list_images_train,
        list_images_val=list_images_val,
        list_images_test=list_images_test,
        sen_size=wandb.config.sen_size,
        use_augmentation=wandb.config.use_augmentation,
        batch_size=wandb.config.batch_size,
        tta_limit=wandb.config.tta_limit
    )
    
    # Initialize the PyTorch Lightning Trainer
    trainer = init_trainer()

    # Train the model
    trainer.fit(model=lightning_model)
    
    # Finish the WandB run
    wandb.finish()
    

def init_wandb():
    """
    Initialize WandB logging and configuration.
    """
    # Initialize WandB with project and entity information
    wandb.init(
        entity='association-rosia',
        project='flair-2',
    )

    # Update config with defaults from a YAML file if no sweep is running
    wandb.config.update(os.path.join('src', 'models', 'config-defaults.yml'))


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
        monitor='val/loss',
        mode='min',
        dirpath=cst.path_models,
        filename=f'{wandb.run.name}-{wandb.run.id}',
        auto_insert_metric_name=False,
        verbose=True
    )

    early_stopping_callback = callbacks.EarlyStopping(
        monitor='val/loss',
        mode='min',
        patience=10,
        min_delta=0,
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
            max_epochs=100,
            logger=loggers.WandbLogger(),
            callbacks=[checkpoint_callback, early_stopping_callback],
            accelerator=cst.device,
            num_sanity_val_steps=0
        )

    return trainer


if __name__ == '__main__':
    main()
