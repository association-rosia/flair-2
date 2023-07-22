import os
from glob import glob
import json

import torch
from torch.utils.data import Dataset

import src.constants as cst
import rasterio


def read_aerial(path_file):
    with rasterio.open(path_file) as f:
        image = f.read()
        image = torch.from_numpy(image)

    return image


def read_centroids(path_file):
    with open(path_file) as f:
        centroids = json.load(f)

    return centroids


class Flair2Dataset(Dataset):
    def __init__(self, is_test):
        self.is_test = is_test
        self.path = cst.PATH_DATA_TRAIN if not is_test else cst.PATH_DATA_TEST
        self.centroids = read_centroids(os.path.join(cst.PATH_DATA, 'centroids_sp_to_patch.json'))
        self.list_images = self.get_list_images()

        # self.path_metadata = os.path.join(path, 'aerial_metadata.json')
        # self.path_aerial = os.path.join(path, 'aerial')
        # self.path_sen = os.path.join(path, 'sen')
        # self.path_labels = os.path.join(path, 'labels')

    def get_list_images(self):
        search_images = os.path.join(self.path, 'aerial', '**', '*.tif')
        list_images = glob(search_images, recursive=True)

        return list_images

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        aerial = read_aerial(self.list_images[idx])

        image_id = self.list_images[idx].split('/')[-1]
        centroid = self.centroids[image_id]
        return None


if __name__ == '__main__':
    dataset_train = Flair2Dataset(is_test=False)
    test = dataset_train[0]
