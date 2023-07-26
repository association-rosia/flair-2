import json
import os
import sys

sys.path.insert(0, os.path.join('.'))
from datetime import datetime
from glob import glob
from tqdm import tqdm

import collections
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.model_selection import train_test_split

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

import src.constants as cst


class FLAIR2Dataset(Dataset):
    def __init__(self, list_images, sen_size, is_test=False, use_augmentation=None):
        self.list_images = list_images
        self.sen_size = sen_size
        self.is_test = is_test
        self.use_augmentation = use_augmentation

        self.path = cst.PATH_DATA_TRAIN if not is_test else cst.PATH_DATA_TEST
        self.path_centroids = os.path.join(cst.PATH_DATA, 'centroids_sp_to_patch.json')
        self.centroids = self.read_centroids(self.path_centroids)

    @staticmethod
    def img_to_mask(path_image):
        path_mask = path_image.replace('aerial', 'labels')
        path_mask = path_mask.replace('img', 'msk')
        path_mask = path_mask.replace('IMG', 'MSK')

        return path_mask

    def get_paths(self, idx):
        path_aerial = self.list_images[idx]
        sen_id = '/'.join(path_aerial.split('/')[-4:-2])
        path_sen = os.path.join(self.path, 'sen', sen_id, 'sen')
        path_labels = self.img_to_mask(path_aerial)
        image_id = path_aerial.split('/')[-1]

        return path_aerial, path_sen, path_labels, image_id

    @staticmethod
    def read_tif(path_file):
        with rasterio.open(path_file) as f:
            image = f.read()
            image = torch.from_numpy(image)
            image = image.type(torch.uint8)  # from 0 to 255

        return image

    @staticmethod
    def read_centroids(path_file):
        with open(path_file) as f:
            centroids = json.load(f)

        return centroids

    def read_sen_npy(self, path_sen, file_name, centroid):
        path_file = os.path.join(path_sen, file_name)
        data = np.load(path_file, mmap_mode='r').astype(np.int16)

        sp_len = int(self.sen_size / 2)
        data = data[:, :, centroid[0] - sp_len:centroid[0] + sp_len, centroid[1] - sp_len:centroid[1] + sp_len]
        data = torch.from_numpy(data)

        return data

    @staticmethod
    def read_sen_txt(path_sen, file_name):
        path_file = os.path.join(path_sen, file_name)
        with open(path_file, 'r') as f:
            sen_products = [line for line in f.readlines()]

        return sen_products

    def read_sens(self, image_id, path_sen):
        centroid = self.centroids[image_id]
        sen_ids = path_sen.split('/')[-3:-1]

        file_data = f'SEN2_sp_{sen_ids[0]}-{sen_ids[1]}_data.npy'
        sen_data = self.read_sen_npy(path_sen, file_data, centroid)

        file_masks = f'SEN2_sp_{sen_ids[0]}-{sen_ids[1]}_masks.npy'
        sen_masks = self.read_sen_npy(path_sen, file_masks, centroid)

        file_products = f'SEN2_sp_{sen_ids[0]}-{sen_ids[1]}_products.txt'
        sen_products = self.read_sen_txt(path_sen, file_products)

        return sen_data, sen_masks, sen_products

    def get_aerial(self, path_aerial):
        aerial = self.read_tif(path_aerial)
        aerial = aerial / 255.0

        return aerial

    @staticmethod
    def extract_sen_months(sen_products):
        sen_dates = [datetime.strptime(product.split('_')[2], '%Y%m%dT%H%M%S') for product in sen_products]
        sen_months = [date.month for date in sen_dates]

        return sen_months

    @staticmethod
    def masks_filtering(sen_data, sen_masks, sen_months, thr_cover=60, thr_rate=0.5):
        times = []

        for t in range(len(sen_masks)):
            # TODO: verify this (cf. filter_dates in ils_dataset.py from FLAIR project)
            cover = np.count_nonzero((sen_masks[t, 0, :, :] >= thr_cover) + (sen_masks[t, 1, :, :] >= thr_cover))
            rate = cover / (sen_masks.shape[2] * sen_masks.shape[3])
            sen_per_months = sen_months.count(sen_months[t])

            if rate < thr_rate or sen_per_months == 1:
                times.append(t)
            else:
                sen_months[t] = -1

        sen_data = sen_data[times, :, :, :]
        sen_months = [sen_month for sen_month in sen_months if sen_month != -1]

        return sen_data, sen_months

    @staticmethod
    def months_averaging(sen_data, sen_months):
        sen_shape = list(sen_data.shape)
        sen_shape[0] = 12
        sen = torch.zeros(sen_shape)

        for m in range(12):
            indexes = [i for i in range(len(sen_months)) if sen_months[i] == m + 1]
            sen[m, :, :, :] = torch.mean(sen_data[indexes, :, :, :].float(), dim=0)

        return sen

    def get_sen(self, image_id, path_sen):
        sen_data, sen_masks, sen_products = self.read_sens(image_id, path_sen)
        sen_months = self.extract_sen_months(sen_products)
        sen_data, sen_months = self.masks_filtering(sen_data, sen_masks, sen_months)
        sen = self.months_averaging(sen_data, sen_months)
        sen = sen / 19779.0  # founded maximum value (minimum = 0)

        return sen

    def get_labels(self, path_labels):
        labels = self.read_tif(path_labels)
        labels = labels - 1

        return labels

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        path_aerial, path_sen, path_labels, image_id = self.get_paths(idx)
        aerial = self.get_aerial(path_aerial)
        sen = self.get_sen(image_id, path_sen)
        labels = self.get_labels(path_labels)
        # TODO: data augmentation and TTA

        return aerial, sen, labels


def get_list_images(path):
    search_images = os.path.join(path, 'aerial', '**', '*.tif')
    list_images = glob(search_images, recursive=True)

    return list_images


if __name__ == '__main__':
    path_train = cst.PATH_DATA_TRAIN
    list_images_train = get_list_images(path_train)

    dataset_train = FLAIR2Dataset(
        list_images=list_images_train,
        sen_size=40,
        is_test=False,
        use_augmentation=False,
    )

    aerial, sen, labels = dataset_train[0]
    print(aerial.shape, sen.shape, labels.shape)
