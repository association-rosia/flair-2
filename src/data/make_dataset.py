import os
from glob import glob
import json

import torch
from torch.utils.data import Dataset

import src.constants as cst
import rasterio

import numpy as np


class FLAIR2Dataset(Dataset):
    def __init__(self, list_images, sen_size, is_test):
        self.list_images = list_images
        self.sen_size = sen_size
        self.is_test = is_test
        self.path = cst.PATH_DATA_TRAIN if not is_test else cst.PATH_DATA_TEST
        self.path_centroids = os.path.join(cst.PATH_DATA, 'centroids_sp_to_patch.json')
        self.centroids = self.read_centroids(self.path_centroids)

    def path_aerial_to_labels(self, path_aerial):
        path_labels = path_aerial.replace('aerial', 'labels')
        path_labels = path_labels.replace('img', 'msk')
        path_labels = path_labels.replace('IMG', 'MSK')

        return path_labels

    def get_paths(self, idx):
        path_aerial = self.list_images[idx]
        sen_id = '/'.join(path_aerial.split('/')[-4:-2])
        path_sen = os.path.join(self.path, 'sen', sen_id, 'sen')
        path_labels = self.path_aerial_to_labels(path_aerial)
        image_id = path_aerial.split('/')[-1]

        return path_aerial, path_sen, path_labels, image_id

    def read_tif(self, path_file):
        with rasterio.open(path_file) as f:
            image = f.read()
            image = torch.from_numpy(image)
            image = image.type(torch.uint8)  # from 0 to 255

        return image

    def read_centroids(self, path_file):
        with open(path_file) as f:
            centroids = json.load(f)

        return centroids

    def read_sen(self, path_sen, file_name, centroid):
        path_data = os.path.join(path_sen, file_name)
        data = np.load(path_data, mmap_mode='r').astype(np.int16)

        sp_len = int(self.sen_size / 2)
        data = data[:, :, centroid[0] - sp_len:centroid[0] + sp_len, centroid[1] - sp_len:centroid[1] + sp_len]
        data = torch.from_numpy(data)

        return data

    def read_sens(self, image_id, path_sen):
        centroid = self.centroids[image_id]
        sen_ids = path_sen.split('/')[-3:-1]

        file_data = f'SEN2_sp_{sen_ids[0]}-{sen_ids[1]}_data.npy'
        sen_data = self.read_sen(path_sen, file_data, centroid)

        file_masks = f'SEN2_sp_{sen_ids[0]}-{sen_ids[1]}_masks.npy'
        sen_masks = self.read_sen(path_sen, file_masks, centroid)

        return sen_data, sen_masks

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        path_aerial, path_sen, path_labels, image_id = self.get_paths(idx)
        aerial = self.read_tif(path_aerial) / 255.0
        sen_data, sen_masks = self.read_sens(image_id, path_sen)
        labels = self.read_tif(path_labels)

        sen = sen_data  # TODO: process using sen_masks
        # TODO: remove data with to many cloud or snow (if > 20)
        # TODO: process labels 1 dim to 13
        # TODO: select sen_data & sen_masks using metadata or products (date)
        # TODO: data augmentation

        return aerial, sen, labels


def get_list_images(path):
    search_images = os.path.join(path, 'aerial', '**', '*.tif')
    list_images = glob(search_images, recursive=True)

    return list_images


if __name__ == '__main__':
    # sen_temp_len_min = 20
    # sen_temp_len_max = 100

    path_train = cst.PATH_DATA_TRAIN
    list_images_train = get_list_images(path_train)
    dataset_train = FLAIR2Dataset(list_images=list_images_train, sen_size=40, is_test=False)
    aerial, sen, labels = dataset_train[0]
    print()
