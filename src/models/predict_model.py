import argparse
import os
import shutil
import sys
from time import time

import torch

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

    def update_variables(self, run_name):
        """
        Create a directory to store predictions for the submission.

        Args:
            run_name (str): Name of the current run.

        Returns:
            path_predictions (str): Path to the directory for storing predictions.
        """
        path_predictions = os.path.join(self.path_submissions, run_name)
        os.makedirs(path_predictions, exist_ok=True)

        return path_predictions

    def load_lightning_model(self, path_run, run_name) -> FLAIR2Lightning:
        """
        Load the trained FLAIR-2 Lightning model checkpoint and configure it for submission.

        Args:
            path_run (str): Path to the directory of the current run.
            run_name (str): Name of the predicted run.

        Returns:
            lightning_model (FLAIR2Lightning): Loaded and configured FLAIR-2 Lightning model.
        """
        lightning_ckpt = os.path.join(self.path_models, f'{run_name}.ckpt')
        lightning_model = FLAIR2Lightning.load_from_checkpoint(lightning_ckpt)
        path_predictions = os.path.join(path_run, 'predictions')
        lightning_model.path_predictions = path_predictions
        os.makedirs(path_predictions, exist_ok=False)

        return lightning_model

    def create_zip_submission(self, run_name, submission_inference_time, path_run):
        """
        Rename the directory containing predictions to confirm the submission.

        Args:
            run_name (str): Name of the current run.
            submission_inference_time (str): Inference time for submission.
            path_run (str): Path to the directory of the current run.

        Returns:
            success (bool): True if renaming is successful, False otherwise.
        """
        name_submission = f'{run_name}_{self.baseline_inference_time}_{submission_inference_time}'
        path_submission = os.path.join(path_run, 'predictions')
        zip_path_submission = os.path.join(self.path_submissions, name_submission)
        shutil.make_archive(zip_path_submission, 'zip', path_submission)
        shutil.rmtree(path_run)

    def __call__(self, run_name):
        """
        Execute the submission process for a given run.

        Args:
            run_name (list): List of run names.

        Returns:
            success (bool): True if the submission process is successful, False otherwise.
        """
        path_run = self.update_variables(run_name)
        lightning_model = self.load_lightning_model(path_run, run_name)

        start = time()
        self.trainer.test(model=lightning_model)
        end = time()

        # 4 seconds is the gap between the displayed time by PL and the calculated time by the librairy "time"
        inference_time_seconds = end - start - 4
        minutes = floor(inference_time_seconds // 60)
        seconds = floor(inference_time_seconds % 60)
        submission_inference_time = f'{minutes}-{seconds}'

        self.create_zip_submission(
            run_name=run_name,
            submission_inference_time=submission_inference_time,
            path_run=path_run
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