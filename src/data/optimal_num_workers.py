import multiprocessing as mp
import os
from time import time

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from make_dataset import FLAIR2Dataset, get_list_images
from src.constants import get_constants

cst = get_constants()

df = pd.read_csv(os.path.join(cst.path_data, 'labels-statistics-12.csv'))
list_images_train = get_list_images(cst.path_data_train)

_, list_images_val = train_test_split(
    list_images_train,
    test_size=0.1,
    random_state=42
)

dataset_val = FLAIR2Dataset(
    list_images=list_images_val,
    sen_size=40,
    sen_temp_size=3,
    sen_temp_reduc='median',
    sen_list_bands=['2', '3', '4', '5', '6', '7', '8', '8a', '11', '12'],
    prob_cover=10,
    is_test=False,
)

for num_workers in range(6, mp.cpu_count(), 2):
    loader_val = DataLoader(
        dataset_val,
        shuffle=True,
        num_workers=num_workers,
        batch_size=24
    )

    start = time()
    for i, data in enumerate(loader_val, 0):
        pass

    end = time()
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))