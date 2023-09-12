from torch.utils.data import DataLoader
from src.data.make_dataset import get_list_images, FLAIR2Dataset
from src.constants import get_constants
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import kstest

cst = get_constants()

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

results = []
for idx, batch in enumerate(dataloader):
    _, _, _, labels = batch
    labels = labels.squeeze()
    plt.imshow(labels)
    plt.show()
    return_counts = labels.unique(return_counts=True)
    return_counts = return_counts[0].tolist(), return_counts[1].tolist()

    distribution = []
    for i in range(12):
        if i in return_counts[0]:
            distribution.append(return_counts[1][return_counts[0].index(i)])
        else:
            distribution.append(0)

    test_statistic, p_value = kstest(distribution, 'uniform')
    results.append((idx, test_statistic, p_value))

# Sort the results by p-value in ascending order
results.sort(key=lambda x: x[2])
# The distribution with the lowest p-value is the closest to uniform
closest_distribution_index = results[0][0]

print(f"The distribution closest to uniform is distribution {closest_distribution_index}")

