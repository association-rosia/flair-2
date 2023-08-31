import os
import sys

import torch

from time import time

sys.path.append('.')

from src.models.lightning import FLAIR2Lightning
import pytorch_lightning as pl

from src.constants import get_constants

cst = get_constants()

torch.set_float32_matmul_precision('high')


class FLAIR2Submission:
    """
    Class for submitting predictions using the FLAIR-2 Lightning model.
    """
    def __init__(self):
        self.nodes = 1
        self.gpus_per_nodes = 1
        self.trainer = pl.Trainer(
            accelerator=cst.device,
            num_nodes=self.nodes,
            # fast_dev_run=3
        )
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

    def load_lightning_model(self, path_run) -> FLAIR2Lightning:
        """
        Load the trained FLAIR-2 Lightning model checkpoint and configure it for submission.

        Args:
            path_run (str): Path to the directory of the current run.

        Returns:
            lightning_model (FLAIR2Lightning): Loaded and configured FLAIR-2 Lightning model.
        """
        lightning_ckpt = os.path.join(self.path_models, f'{run_name}.ckpt')
        lightning_model = FLAIR2Lightning.load_from_checkpoint(lightning_ckpt)
        path_predictions = os.path.join(path_run, 'not_confirmed')
        lightning_model.path_predictions = path_predictions
        os.makedirs(path_predictions, exist_ok=False)

        return lightning_model

    def rename_submissions_dir(self, run_name, submission_inference_time, path_run):
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
        new_path_submission = os.path.join(path_run, name_submission)
        old_path_submission = os.path.join(path_run, 'not_confirmed')
        os.rename(old_path_submission, new_path_submission)

        return os.path.exists(new_path_submission)

    def __call__(self, run_name):
        """
        Execute the submission process for a given run.

        Args:
            run_name (str): Name of the current run.

        Returns:
            success (bool): True if the submission process is successful, False otherwise.
        """
        path_run = self.update_variables(run_name)
        lightning_model = self.load_lightning_model(path_run)

        start = time()
        self.trainer.test(model=lightning_model)
        end = time()

        # inference_time_seconds = (starter.elapsed_time(ender) / 1000.0) * (self.nodes * self.gpus_per_nodes)
        inference_time_seconds = end - start
        submission_inference_time = f'{inference_time_seconds // 60}-{inference_time_seconds % 60}'

        return self.rename_submissions_dir(
            run_name=run_name,
            submission_inference_time=submission_inference_time,
            path_run=path_run
        )


if __name__ == '__main__':
    sub = FLAIR2Submission()
    run_name = 'olive-bird-4-u9k58i02'
    sub(run_name=run_name)
