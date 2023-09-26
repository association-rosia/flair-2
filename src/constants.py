import os

from torch import cuda


class FLAIR2Constants:
    """
    Class to define constants and configurations for the FLAIR-2 project.
    """

    def __init__(self) -> None:
        # Paths to data directories
        self.path_data = os.path.join('data', 'raw')
        self.path_data_train = os.path.join(self.path_data, 'train')
        self.path_data_test = os.path.join(self.path_data, 'test')

        # Paths for models and submissions
        self.path_models = 'models'
        self.path_submissions = 'submissions'

        # Placeholder for baseline inference time (needs to be defined)
        self.baseline_inference_time = '5-57'  # max 14-52

        # Number of worker threads for data loading
        self.train_num_workers = 4
        self.test_num_workers = 10  # same as in the baseline

        # Initialize the device for computation
        self.device = self.init_device()

        sen_band = ['2', '3', '4', '5', '6', '7', '8', '8a', '11', '12']
        self.sen_band2idx = {band: i for i, band in enumerate(sen_band)}

        aerial_band = ['R', 'G', 'B', 'NIR', 'DSM']
        self.aerial_band2idx = {band: i for i, band in enumerate(aerial_band)}

    @staticmethod
    def init_device():
        """
        Initialize device to work with.
        
        Returns:
            device (str): Device to work with (cpu, gpu).
        """
        device = 'cpu'
        if cuda.is_available():
            device = 'gpu'
        # * MPS is not implemented
        # elif mps.is_available():
        #     device = 'mps'

        return device


def get_constants() -> FLAIR2Constants:
    """
    Get an instance of FLAIR2Constants class with predefined constants and configurations.

    Returns:
        FLAIR2Constants: Instance of FLAIR2Constants class.
    """
    return FLAIR2Constants()
