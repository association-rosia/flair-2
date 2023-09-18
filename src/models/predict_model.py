import argparse
import os
import shutil
import sys
from time import time
from glob import glob
import tifffile as tiff

import torch
import numpy as np

sys.path.append('.')

from src.models.lightning import FLAIR2Lightning
import pytorch_lightning as pl



from src.constants import get_constants

cst = get_constants()

from math import floor

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning.trainer.connectors.logger_connector"
                                                               ".logger_connector")

torch.set_float32_matmul_precision('high')


class FLAIR2Submission:
    """
    Class for submitting predictions using the FLAIR-2 Lightning model.
    """

    def __init__(self):
        self.trainer = pl.Trainer(accelerator=cst.device)
        self.baseline_inference_time = cst.baseline_inference_time
        self.path_models = cst.path_models
        self.path_submissions = cst.path_submissions

    def create_path_predictions(self, dir_predictions):
        """
        Create a directory to store predictions for the submission.

        Args:
            dir_predictions (str): Name of the current run.

        Returns:
            path_predictions (str): Path to the directory for storing predictions.
        """
        path_predictions = os.path.join(self.path_submissions, dir_predictions)
        
        if os.path.exists(path_predictions) and not len(os.listdir(path_predictions)) == 0:
            raise FileExistsError('Directory exist and it is not empty.')
        
        os.makedirs(path_predictions, exist_ok=True)

        return path_predictions

    def load_lightning_model(self, path_predictions, name_run, assemble) -> FLAIR2Lightning:
        """
        Load the trained FLAIR-2 Lightning model checkpoint and configure it for submission.

        Args:
            path_run (str): Path to the directory of the current run.
            run_name (str): Name of the predicted run.

        Returns:
            lightning_model (FLAIR2Lightning): Loaded and configured FLAIR-2 Lightning model.
        """
        lightning_ckpt = os.path.join(self.path_models, f'{name_run}.ckpt')
        lightning_model = FLAIR2Lightning.load_from_checkpoint(lightning_ckpt)
        lightning_model.path_predictions = path_predictions
        lightning_model.assemble = assemble

        return lightning_model

    def create_zip_submission(self, dir_predictions, inference_time_seconds):
        """
        Rename the directory containing predictions to confirm the submission.

        Args:
            run_name (str): Name of the current run.
            submission_inference_time (str): Inference time for submission.
            path_run (str): Path to the directory of the current run.

        Returns:
            success (bool): True if renaming is successful, False otherwise.
        """
        minutes = floor(inference_time_seconds // 60)
        seconds = floor(inference_time_seconds % 60)
        submission_inference_time = f'{minutes}-{seconds}'

        path_predictions = os.path.join(self.path_submissions, dir_predictions)
        name_submission = f'{dir_predictions}_{self.baseline_inference_time}_{submission_inference_time}'
        path_submission_zip = os.path.join(self.path_submissions, name_submission)
        shutil.make_archive(path_submission_zip, 'zip', path_predictions)
        shutil.rmtree(path_predictions)
        
    def unique_submission(self, name_run):
        path_predictions = self.create_path_predictions(dir_predictions=name_run)
        lightning_model = self.load_lightning_model(
            path_predictions=path_predictions, 
            name_run=name_run, 
            assemble=False
        )

        start = time()
        self.trainer.test(model=lightning_model)
        end = time()
        
        # 4 seconds is the gap between the displayed time by PL and the calculated time by the librairy "time"
        return name_run, end - start - 4
        
    def assemble_submission(self, name_runs):
        dir_predictions = '_'.join(name_runs)
        dir_tensors = dir_predictions + '_tensor'
        
        path_predictions = self.create_path_predictions(dir_predictions)
        path_tensors = self.create_path_predictions(dir_tensors)
        
        start = time()
        for name_run in name_runs:
            lightning_model = self.load_lightning_model(
                path_predictions=path_tensors,
                name_run=name_run, 
                assemble=True
            )
            
            self.trainer.test(model=lightning_model)
        
        path_template = os.path.join(path_tensors, '*.pt')
        for path_tensor in glob(path_template):
            pred_label = torch.load(path_tensor)
            
            img = pred_label.argmax(dim=0)
            img = pred_label.numpy(force=True)
            img = img.astype(dtype=np.uint8)
            
            name_img = os.path.basename(path_tensor)
            id_img = os.path.splitext(name_img)[0]
            path_img = os.path.join(path_predictions, f'PRED_{id_img}')
            
            tiff.imwrite(path_img, img, dtype=np.uint8, compression='LZW')

        end = time()
        
        shutil.rmtree(path_tensors)
        
        # 4 seconds is the gap between the displayed time by PL and the calculated time by the librairy "time"
        return dir_predictions, end - start - 4

    def __call__(self, name_runs):
        """
        Execute the submission process for a given run.

        Args:
            run_name (list | str): List of run names.

        Returns:
            success (bool): True if the submission process is successful, False otherwise.
        """
        
        if isinstance(name_runs, list) and len(name_runs) >= 2:
            dir_predictions, inference_time_seconds = self.assemble_submission(name_runs)
        else:
            dir_predictions, inference_time_seconds = self.unique_submission(name_runs)
 
        self.create_zip_submission(
            dir_predictions=dir_predictions,
            inference_time_seconds=inference_time_seconds,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for creating submissions with specified models names. Unique name for classique inference or multiple names for assemble.')
    parser.add_argument('-n', '--name', nargs='+', type=str, help='Name of models to use for submissions')
    args = parser.parse_args()

    sub = FLAIR2Submission()
    if args.name:
        sub(run_name=args.name)
    else:
        print('Please provide a model name using the -n or --name parameter.')