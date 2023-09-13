import os
import sys

sys.path.append(os.curdir)

from statistics import mean
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.constants import get_constants
from src.data.make_dataset import get_list_images, FLAIR2Dataset

from scipy import special

cst = get_constants()

import pandas as pd

path_data = cst.path_data_train
list_images_train = get_list_images(path_data)

list_images_train, list_images_val = train_test_split(list_images_train,
                                                      test_size=0.1,
                                                      random_state=42)

dataset = FLAIR2Dataset(
    list_images=list_images_val,
    sen_size=40,
    sen_temp_size=3,
    sen_temp_reduc='median',
    sen_list_bands=['2', '3', '4', '5', '6', '7', '8', '8a', '11', '12'],
    prob_cover=10,
    is_test=False,
)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=1,
    shuffle=False,
)

df = pd.read_csv('data/raw/labels-statistics-12.csv')
target_distribution = list(df['Freq.-test (%)'])

max_number_classes = 0
for idx, batch in enumerate(dataloader):
    _, _, _, labels = batch
    # get the unique values and the associated counts
    return_counts = labels.unique(return_counts=True)

    # convert to list
    return_counts = return_counts[0].tolist(), return_counts[1].tolist()

    max_number_classes = max(max_number_classes, len(return_counts[0]))
    print(f'Testing {idx} - Max number of classes = {max_number_classes}')

print()
results = []
for idx, batch in enumerate(dataloader):
    print(f'Testing {idx}')
    _, _, _, labels = batch

    num_total_pixels = labels.shape[-2] * labels.shape[-1]

    # get the unique values and the associated counts
    return_counts = labels.unique(return_counts=True)

    # convert to list
    return_counts = return_counts[0].tolist(), [num_pixels / num_total_pixels for num_pixels in return_counts[1].tolist()]

    if len(return_counts[0]) == max_number_classes:
        # build the distribution with all classes possible values
        distribution = []
        for i in range(13):
            if i in return_counts[0]:
                distribution.append(return_counts[1][return_counts[0].index(i)])
            else:
                distribution.append(0)

        # compute statistical test to determine how close the distribution is close to uniform
        kl_div = special.kl_div(distribution, target_distribution)
        kl_div = mean(kl_div)
        results.append((idx, kl_div))

# Sort the results by p-value in ascending order
results.sort(key=lambda x: x[1])
# The distribution with the lowest p-value is the closest to uniform
closest_distribution_index = results[0][0]

print(f"The distribution closest to uniform is distribution {closest_distribution_index}")
