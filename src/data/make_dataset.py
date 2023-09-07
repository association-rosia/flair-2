import json
import os
import sys

sys.path.insert(0, os.path.join('.'))
from datetime import datetime
from glob import glob

import numpy as np
import rasterio
import math
import torch
from torch.utils.data import Dataset

from src.constants import get_constants

cst = get_constants()


class FLAIR2Dataset(Dataset):
    def __init__(self, list_images, sen_size, is_test=False):
        self.sen_min = 1000
        self.sen_max = 0

        self.list_images = list_images
        self.sen_size = sen_size
        self.is_test = is_test

        self.path = cst.path_data_train if not self.is_test else cst.path_data_test
        self.path_centroids = os.path.join(cst.path_data, 'centroids_sp_to_patch.json')
        self.centroids = self.read_centroids(self.path_centroids)

    @staticmethod
    def img_to_msk(path_image):
        path_mask = path_image.replace('aerial', 'labels')
        path_mask = path_mask.replace('img', 'msk')
        path_mask = path_mask.replace('IMG', 'MSK')

        return path_mask

    def get_paths(self, idx):
        path_aerial = self.list_images[idx]
        path_aerial = os.path.normpath(path_aerial)
        list_dir_aerial = path_aerial.split(os.sep)[-4:-2]
        path_sen = os.path.join(self.path, 'sen', *list_dir_aerial, 'sen')
        path_labels = self.img_to_msk(path_aerial)
        name_image = os.path.basename(path_aerial)

        return path_aerial, path_sen, path_labels, name_image

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

    def read_sens(self, name_image, path_sen):
        centroid = self.centroids[name_image]
        path_sen = os.path.normpath(path_sen)
        sen_ids = path_sen.split(os.sep)[-3:-1]

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
    def extract_channels(sen_products, num_channels):
        sen_dates = [datetime.strptime(product.split('_')[2], '%Y%m%dT%H%M%S') for product in sen_products]
        sen_channels = [math.ceil(date.month / (12 / num_channels)) for date in sen_dates]

        return sen_channels

    # @staticmethod
    # def masks_filtering(sen_data, sen_masks, sen_months, thr_cover=60, thr_rate=0.5):
    #     times = []
    #
    #     for t in range(len(sen_masks)):
    #         cover = np.count_nonzero((sen_masks[t, 0, :, :] >= thr_cover) + (sen_masks[t, 1, :, :] >= thr_cover))
    #         rate = cover / (sen_masks.shape[2] * sen_masks.shape[3])
    #         sen_per_months = sen_months.count(sen_months[t])
    #
    #         if rate < thr_rate or sen_per_months == 1:
    #             times.append(t)
    #         else:
    #             sen_months[t] = -1
    #
    #     sen_data = sen_data[times, :, :, :]
    #     sen_months = [sen_month for sen_month in sen_months if sen_month != -1]
    #
    #     return sen_data, sen_months

    @staticmethod
    def channels_averaging(sen_data, sen_masks, sen_channels, num_channels, prob_cover=50):
        sen_shape = list(sen_data.shape)
        sen_shape[0] = num_channels
        sen = torch.zeros(sen_shape)

        for channel in range(num_channels):
            idx = [i for i in range(len(sen_channels)) if sen_channels[i] == channel + 1]
            cover = (sen_masks[idx, 0, :, :] >= prob_cover) + (sen_masks[idx, 1, :, :] >= prob_cover)
            cover = cover.unsqueeze(1).repeat(1, 10, 1, 1)
            sen_nan = torch.where(cover, float('nan'), sen_data[idx, :, :, :])
            sen[channel, :, :, :] = torch.nanmean(sen_nan, dim=0)

        sen = torch.where(torch.isnan(sen), 0, sen)

        return sen

    def get_sen(self, name_image, path_sen, num_channels=3):
        sen_data, sen_masks, sen_products = self.read_sens(name_image, path_sen)
        sen_channels = self.extract_channels(sen_products, num_channels)
        sen = self.channels_averaging(sen_data, sen_masks, sen_channels, num_channels)
        sen = sen / 18172.0  # computed maximum value (minimum = 0)

        # uncomment to compute the maximum value
        # self.sen_min = min(torch.min(sen).item(), self.sen_min)
        # self.sen_max = max(torch.max(sen).item(), self.sen_max)
        # print(self.sen_min, self.sen_max)

        return sen

    def get_labels(self, path_labels):
        labels = self.read_tif(path_labels)
        labels = labels - 1
        labels = torch.where(labels < 13, labels, 12)

        return torch.squeeze(labels)

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        path_aerial, path_sen, path_labels, name_image = self.get_paths(idx)
        aerial = self.get_aerial(path_aerial)
        sen = self.get_sen(name_image, path_sen)

        if self.is_test:
            labels = torch.ByteTensor()
        else:
            labels = self.get_labels(path_labels)

        return name_image, aerial, sen, labels


def get_list_images(path):
    search_images = os.path.join(path, 'aerial', '**', '*.tif')
    list_images = glob(search_images, recursive=True)

    return list_images


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    path_data = cst.path_data_train
    list_images = get_list_images(path_data)

    dataset = FLAIR2Dataset(
        list_images=list_images,
        sen_size=40,
        is_test=False,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
    )

    image_id, aerial, sen, labels = dataset[0]
    print(image_id, aerial.shape, sen.shape, labels.shape)

    for image_id, aerial, sen, labels in dataloader:
        # print(image_id, aerial.shape, sen.shape, labels.shape)
        # break
        pass
