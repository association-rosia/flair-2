import os
from torch import cuda
from torch.backends import mps 

class FLAIR2Constants():
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
        self.baseline_inference_time = 'a_definir'
        
        # Number of worker threads for data loading
        self.num_workers = 8
        
        # Initialize the device for computation
        self.device = self.init_device()
        
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
