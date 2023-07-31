import os

class FLAIR2Constants():
    path_data = os.path.join('data', 'raw')
    path_data_train = os.path.join(path_data, 'train')
    path_data_test = os.path.join(path_data, 'test')

    path_models = 'models'

    path_submissions = 'submissions'
    
    bl_inf_time = ''
    
    
def get_constants() -> FLAIR2Constants:
    return FLAIR2Constants()


print(get_constants())