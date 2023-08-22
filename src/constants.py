import os


class FLAIR2Constants():
    path_data = os.path.join('data', 'raw')
    path_data_train = os.path.join(path_data, 'train')
    path_data_test = os.path.join(path_data, 'test')
    path_models = 'models'
    path_submissions = 'submissions'
    baseline_inference_time = 'a_definir'
    num_workers = 8


def get_constants() -> FLAIR2Constants:
    return FLAIR2Constants()
