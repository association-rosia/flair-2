import os


class FLAIR2Constants():
    def __init__(self) -> None:
        self.path_data = os.path.join('data', 'raw')
        self.path_data_train = os.path.join(self.path_data, 'train')
        self.path_data_test = os.path.join(self.path_data, 'test')
        self.path_models = 'models'
        self.path_submissions = 'submissions'
        self.baseline_inference_time = 'a_definir'
        self.num_workers = 8


def get_constants() -> FLAIR2Constants:
    return FLAIR2Constants()
