import os
import sys

sys.path.append('.')

from src.models.lightning import FLAIR2Lightning
import pytorch_lightning as pl

from src.constants import get_constants

cst = get_constants()


class FLAIR2Submission():
    def __init__(self):
        self.nodes = 1
        self.gpus_per_nodes = 1
        self.trainer = pl.Trainer(
            accelerator='gpu',
            # accelerator='cpu',
            num_nodes=self.nodes,
            fast_dev_run=3
        )
        self.baseline_inference_time = cst.baseline_inference_time
        self.path_models = cst.path_models
        self.path_submissions = cst.path_submissions

    def update_variables(self, run_name):
        path_predictions = os.path.join(self.path_submissions, run_name)
        os.makedirs(path_predictions, exist_ok=True)

        return path_predictions

    def load_lightning_model(self, path_run) -> FLAIR2Lightning:
        lightning_ckpt = os.path.join(self.path_models, f'{run_name}.ckpt')
        lightning_model = FLAIR2Lightning.load_from_checkpoint(lightning_ckpt)
        path_predictions = os.path.join(path_run, 'not_confirmed')
        lightning_model.path_predictions = path_predictions
        os.makedirs(path_predictions, exist_ok=False)

        return lightning_model

    def rename_submissions_dir(self, run_name, submission_inference_time, path_run):
        # name_of_your_approach = f'{lightning_model.architecture}-{lightning_model.encoder_name}'
        name_of_your_approach = run_name
        name_submission = f'{name_of_your_approach}_{self.baseline_inference_time}_{submission_inference_time}'
        new_path_submission = os.path.join(path_run, name_submission)
        old_path_submission = os.path.join(path_run, 'not_confirmed')
        os.rename(old_path_submission, new_path_submission)

        return os.path.exists(new_path_submission)

    def __call__(self, run_name):
        path_run = self.update_variables(run_name)
        lightning_model = self.load_lightning_model(path_run)

        # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # starter.record()

        self.trainer.test(model=lightning_model)

        # dist.barrier() # ensures synchronization among distributed processes
        # torch.cuda.synchronize() # ensures synchronization between the CPU and GPU
        # ender.record() # end time

        # inference_time_seconds = (starter.elapsed_time(ender) / 1000.0) * (self.nodes * self.gpus_per_nodes)
        inference_time_seconds = 100000
        submission_inference_time = f'{inference_time_seconds // 60}-{inference_time_seconds % 60}'

        return self.rename_submissions_dir(
            run_name=run_name,
            submission_inference_time=submission_inference_time,
            path_run=path_run
        )


if __name__ == '__main__':
    sub = FLAIR2Submission()
    run_name = 'sandy-night-27-krtk1bo0'
    sub(run_name=run_name)
