import os
import sys
import datetime as dt

sys.path.append('.')

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from src.data.make_dataset import FLAIR2Dataset, get_list_images

from src.models.lightning import FLAIR2Lightning
import pytorch_lightning as pl

import src.constants as cst


def main():
    os.makedirs(cst.PATH_SUBMISSIONS, exist_ok=True)
    
    list_images = get_list_images(cst.PATH_DATA_TEST)
    dataset = FLAIR2Dataset(
        list_images=list_images,
        sen_size=40,
        is_test=False,
        use_augmentation=False,
    )
    
    dataloader_eval = DataLoader(
        dataset=dataset,
        batch_size=16, 
        shuffle=False, 
        drop_last=False
    )
    
    model_name = "amber-morning-43-ibxuw5qk.ckpt"
    lightning_model = FLAIR2Lightning.load_from_checkpoint(
        os.path.join(cst.PATH_MODELS, model_name),
        )

    nodes = 1
    gpus_per_nodes = 1
    trainer = pl.Trainer(
        # accelerator='gpu',
        accelerator='cpu',
        num_nodes=nodes,
        limit_train_batches=3,
        limit_val_batches=3,
        limit_test_batches=3,
    )
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record() # start time
    
    trainer.predict(
        model=lightning_model,
        dataloaders=dataloader_eval,
        return_predictions=False
    )
    
    dist.barrier() # ensures synchronization among distributed processes
    torch.cuda.synchronize() # ensures synchronization between the CPU and GPU
    ender.record() # end time

    
    inference_time_seconds = (starter.elapsed_time(ender) / 1000.0) * (nodes * gpus_per_nodes)
    
    name_of_your_approach = f"{lightning_model.architecture}_{lightning_model.encoder_name}"
    baseline_inference_time = "MM-S"
    submission_inference_time = f"{inference_time_seconds // 60}-{inference_time_seconds % 60}"
    folder_name = f"{name_of_your_approach}_{baseline_inference_time}_{submission_inference_time}"
    
    
    
if __name__ == '__main__':
    main()