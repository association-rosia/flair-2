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

import ttach as tta

from src.constants import get_constants

cst = get_constants()


class FLAIR2Submission():
    def __init__(self):
        self.nodes = 1
        self.gpus_per_nodes = 1
        self.trainer = pl.Trainer(
            # accelerator='gpu',
            accelerator='cpu',
            num_nodes=self.nodes,
            fast_dev_run=3
        )
        self.baseline_inference_time = cst.baseline_inference_time
        
        self.lightning_ckpt = None
        self.path_submissions = None
        
    def update_variables(self, run_name):
        self.lightning_ckpt = f'{run_name}.ckpt'
        self.path_submissions = os.path.join(cst.path_submissions, run_name)
        os.makedirs(self.path_submissions, exist_ok=False)
        
    def reset_variables(self):
        self.lightning_ckpt = None
        self.path_submissions = None
        
    def load_lightning_model(self, apply_tta)->FLAIR2Lightning:
        lightning_model = FLAIR2Lightning.load_from_checkpoint(self.lightning_ckpt)
        lightning_model.apply_tta = apply_tta
        lightning_model.path_submissions = os.path.join(self.path_submissions, 'not_confirmed')
        os.makedirs(self.path_submissions, exist_ok=False)
        return lightning_model
    
    def __call__(self, run_name, apply_tta):
        self.run_name = run_name
        self.update_variables(run_name)
        lightning_model = self.load_lightning_model(apply_tta)
        
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
    
        self.trainer.test(
            model=lightning_model,
            return_predictions=False
        )
    
        dist.barrier() # ensures synchronization among distributed processes
        torch.cuda.synchronize() # ensures synchronization between the CPU and GPU
        ender.record() # end time
        
        inference_time_seconds = (starter.elapsed_time(ender) / 1000.0) * (self.nodes * self.gpus_per_nodes)
        submission_inference_time = f'{inference_time_seconds // 60}-{inference_time_seconds % 60}'
        name_of_your_approach = f'{lightning_model.architecture}-{lightning_model.encoder_name}'
        name_of_your_approach = 'tta-' + name_of_your_approach if apply_tta else name_of_your_approach
        name_submission = f'{name_of_your_approach}_{self.baseline_inference_time}_{submission_inference_time}'
        new_path_submission = os.path.join(self.path_submissions, name_submission)
        old_path_submission = os.path.join(self.path_submissions, 'not_confirmed')
        os.rename(old_path_submission, new_path_submission)
        
        self.reset_variables()
        
        return True
    
if __name__ == '__main__':
    run_name = 'amber-morning-43-ibxuw5qk'
    sub = FLAIR2Submission()
    sub(run_name=run_name, apply_tta=False)
    sub(run_name=run_name, apply_tta=True)