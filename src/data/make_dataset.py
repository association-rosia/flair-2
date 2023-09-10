import json
import os
import sys
import time

sys.path.append(os.curdir)

from datetime import datetime
from glob import glob
from tqdm import tqdm

import math
import numpy as np
import tifffile
import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms as T

from src.constants import get_constants

cst = get_constants()


class FLAIR2Dataset(Dataset):
    def __init__(
            self, 
            list_images: list[str], 
            sen_size: int,
            sen_temp_size: int,
            sen_temp_reduc: str,
            sen_list_bands: list[str],
            prob_cover: int,
            is_test: bool,
        ):
        """
        Initialize the FLAIR2Dataset.

        Args:
            list_images (list[str]): List of image paths.
            sen_size (int): Size of sentinel images.
            sen_temp_size (int): Size of temporal channel for Sentinel 2 images.
            sen_temp_reduc (str): Temporal data reduction method.
            sen_list_bands (list[str]): List of sentinel bands.
            prob_cover (int): Probability cover value.
            is_test (bool): Indicates if the dataset is for testing.
        """
        self.list_images = list_images
        
        if sen_size <= 0:
            raise ValueError(f'sen_size is a size, it must be positif but found {sen_size}')
        self.prob_cover = prob_cover
        self.sen_size = sen_size
        self.sen_temp_size = sen_temp_size
        
        possible_reduction = ['median', 'mean']
        if not sen_temp_reduc in possible_reduction:
            raise ValueError(f'sen_temp_reduc can be on of {", ".join(possible_reduction)} but found {sen_temp_reduc}.')
        self.sen_temp_reduc = sen_temp_reduc
        
        dataset_band = ['2', '3', '4', '5', '6', '7', '8', '8a', '11', '12']
        for band in sen_list_bands:
            if not band in dataset_band:
                raise ValueError(f'sen_list_bands can be composed of {", ".join(dataset_band)} but found {band}.')
        self.band2idx = {band: i for i, band in enumerate(dataset_band)}
        self.sen_list_bands = sen_list_bands
        self.sen_idx_band = [self.band2idx[str(band)] for band in self.sen_list_bands]
        
        if prob_cover <= 0:
            raise ValueError(f'prob_cover is an integer between 1 and 100, found {prob_cover}')
        self.prob_cover = prob_cover
        self.is_test = is_test

        self.path = cst.path_data_train if not self.is_test else cst.path_data_test
        self.path_centroids = os.path.join(cst.path_data, 'centroids_sp_to_patch.json')
        self.centroids = self.read_centroids(self.path_centroids)
        
        self.aerial_normalize = self.init_aerial_normalize()
        
        self.sen_normalize = lambda x: x
        self.sen_normalize = self.init_sen_normalize()
        

    @staticmethod
    def img_to_msk(path_image: str) -> str:
        """
        Convert an image path to a mask path.

        Args:
            path_image (str): Input image path.

        Returns:
            str: Mask path.
        """
        path_mask = path_image.replace('aerial', 'labels')
        path_mask = path_mask.replace('img', 'msk')
        path_mask = path_mask.replace('IMG', 'MSK')

        return path_mask

    def get_paths(self, idx) -> tuple[str, str, str, str]:
        """
        Get paths for aerial image, sentinel image, labels, and image name.

        Args:
            idx (int): Index of the dataset.

        Returns:
            tuple[str, str, str, str]: Paths for aerial, sentinel, labels, and image name.
        """
        path_aerial = self.list_images[idx]
        path_aerial = os.path.normpath(path_aerial)
        list_dir_aerial = path_aerial.split(os.sep)[-4:-2]
        path_sen = os.path.join(self.path, 'sen', *list_dir_aerial, 'sen')
        path_labels = self.img_to_msk(path_aerial)
        name_image = os.path.basename(path_aerial)

        return path_aerial, path_sen, path_labels, name_image
    
    @staticmethod
    def init_aerial_normalize() -> T.Compose:
        """
        Initialize aerial image normalization.

        Returns:
            T.Compose: Aerial image normalization transform.
        """
        path_aerial_pixels_metadata = os.path.join(cst.path_data, 'aerial_pixels_metadata.json')
        with open(path_aerial_pixels_metadata, 'r') as f:
            stats = json.load(f)
        
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=stats['mean'], std=stats['std'])    
        ])
    
    def compute_sen_mean(
            self,
            list_path_aerial: list[str],
            total_band_pixel: int,
            sen_idx_band: list[int],
            num_bands: int
        ) -> Tensor:
        """
        Compute the mean of sentinel bands.

        Args:
            list_path_aerial (list[str]): List of aerial image paths.
            total_band_pixel (int): Total number of band pixels.
            sen_idx_band (list[int]): List of sentinel band indices.
            num_bands (int): Number of bands.

        Returns:
            Tensor: Mean of sentinel bands.
        """
        band_sum = torch.zeros(num_bands)
        
        for path_aerial in tqdm(list_path_aerial, desc='Compute sentinel bands mean'):
            path_aerial = os.path.normpath(path_aerial)
            list_dir_aerial = path_aerial.split(os.sep)[-4:-2]
            path_sen = os.path.join(self.path, 'sen', *list_dir_aerial, 'sen')
            name_aerial = os.path.basename(path_aerial)
            sen_data = self.get_sen(name_aerial, path_sen, sen_idx_band)
            
            band_sum += sen_data.sum(dim=[0, 2, 3])
        
        return band_sum / total_band_pixel
    
    def compute_sen_std(
            self, 
            list_path_aerial: list[str],
            total_band_pixel: int,
            sen_idx_band: list[int],
            num_bands: int,
            band_mean: Tensor
        ) -> Tensor:
        """
        Compute the standard deviation of sentinel bands.

        Args:
            list_path_aerial (list[str]): List of aerial image paths.
            total_band_pixel (int): Total number of band pixels.
            sen_idx_band (list[int]): List of sentinel band indices.
            num_bands (int): Number of bands.
            band_mean (Tensor): Mean of sentinel bands.

        Returns:
            Tensor: Standard deviation of sentinel bands.
        """
        band_mean_squared_diff_sum = torch.zeros(num_bands)
        band_mean = band_mean.view(1, num_bands, 1, 1)
        
        for path_aerial in tqdm(list_path_aerial, desc='Compute sentinel bands std'):
            path_aerial = os.path.normpath(path_aerial)
            list_dir_aerial = path_aerial.split(os.sep)[-4:-2]
            path_sen = os.path.join(self.path, 'sen', *list_dir_aerial, 'sen')
            name_aerial = os.path.basename(path_aerial)
            sen_data = self.get_sen(name_aerial, path_sen, sen_idx_band)
            
            sen_diff = sen_data - band_mean
            band_mean_squared_diff_sum += (sen_diff * sen_diff).sum(dim=[0, 2, 3])
        
        return torch.sqrt(band_mean_squared_diff_sum / total_band_pixel)
    
    def compute_sen_stats(self) -> tuple[Tensor, Tensor]:
        """
        Compute sentinel statistics (mean and standard deviation).

        Returns:
            tuple[Tensor, Tensor]: Mean and standard deviation of sentinel bands.
        """
        list_path_aerial = glob(os.path.join(self.path, 'aerial', '**', '*.tif'), recursive=True)
        num_bands = len(self.sen_list_bands)
        
        total_band_pixel = self.sen_temp_size * self.sen_size * self.sen_size * len(list_path_aerial)
        sen_idx_band = list(self.band2idx.values())
        
        band_mean = self.compute_sen_mean(list_path_aerial, total_band_pixel, sen_idx_band, num_bands)
        band_std = self.compute_sen_std(list_path_aerial, total_band_pixel, sen_idx_band, num_bands, band_mean)
        
        return band_mean, band_std
    
    @staticmethod
    def load_sen_metadata(path_sen_metadata):
        """
        Load sentinel metadata from a file.

        Args:
            path_sen_metadata (str): Path to the sentinel metadata file.

        Returns:
            list: Loaded sentinel metadata.
        """
        if not os.path.exists(path_sen_metadata):
            sen_metadata = []
        else:
            with open(path_sen_metadata, 'r') as f:
                sen_metadata = json.load(f)
        
        return sen_metadata
    
    @staticmethod
    def update_sen_metadata(sen_metadata, path_sen_metadata):
        """
        Update sentinel metadata and save it to a file.

        Args:
            sen_metadata (list): Sentinel metadata to be saved.
            path_sen_metadata (str): Path to the sentinel metadata file.
        """
        with open(path_sen_metadata, 'w') as f:
            json.dump(sen_metadata, f)
        
    def init_sen_normalize(self) -> T.Compose:
        """
        Initialize sentinel image normalization.

        Returns:
            T.Compose: Sentinel image normalization transform.
        """
        path_sen_metadata = os.path.join(cst.path_data, f'sen_pixels_metadata.json')
        sen_metadata = self.load_sen_metadata(path_sen_metadata)
        
        band_mean = None
        band_std = None
        is_good = False
        
        for sen_stats in sen_metadata:
            is_good = (
                sen_stats['image_size'] == self.sen_size and
                sen_stats['temporal_reduction'] == self.sen_temp_reduc and
                sen_stats['temporal_size'] == self.sen_temp_size and
                sen_stats['cover_probability'] == self.prob_cover
            )
            if is_good:
                band_mean = torch.Tensor(sen_stats['mean'])
                band_std = torch.Tensor(sen_stats['std'])
        
        if not is_good:
            band_mean, band_std = self.compute_sen_stats()
            sen_metadata.append({
                'image_size': self.sen_size,
                'temporal_reduction': self.sen_temp_reduc,
                'temporal_size': self.sen_temp_size,
                'cover_probability': self.prob_cover,
                'mean': band_mean.tolist(),
                'std': band_std.tolist()
            })
            self.update_sen_metadata(sen_metadata, path_sen_metadata)
            
        return T.Compose([T.Normalize(
            mean=band_mean[self.sen_idx_band], std=band_std[self.sen_idx_band]
            )])

    @staticmethod
    def read_centroids(path_file):
        """
        Read centroids from a file.

        Args:
            path_file (str): Path to the centroids file.

        Returns:
            dict: Centroids data.
        """
        with open(path_file) as f:
            centroids = json.load(f)

        return centroids

    def read_sen_npy(self, path_sen, file_name, centroid):
        """
        Read sentinel data from a .npy file.

        Args:
            path_sen (str): Path to the sentinel data directory.
            file_name (str): Name of the .npy file.
            centroid (tuple): Centroid coordinates.

        Returns:
            Tensor: Loaded sentinel data.
        """
        path_file = os.path.join(path_sen, file_name)
        data = np.load(path_file, mmap_mode='r').astype(np.int16)

        sp_len = int(self.sen_size / 2)
        data = data[:, :, centroid[0] - sp_len:centroid[0] + sp_len, centroid[1] - sp_len:centroid[1] + sp_len]
        data = torch.from_numpy(data)

        return data

    @staticmethod
    def read_sen_txt(path_sen: str, file_name: str) -> list[str]:
        """
        Read sentinel products from a .txt file.

        Args:
            path_sen (str): Path to the sentinel data directory.
            file_name (str): Name of the .txt file.

        Returns:
            list[str]: List of sentinel products.
        """
        path_file = os.path.join(path_sen, file_name)
        with open(path_file, 'r') as f:
            sen_products = [line for line in f.readlines()]

        return sen_products

    def read_sens(self, name_image, path_sen) -> tuple[Tensor, Tensor, list[str]]:
        """
        Read sentinel data, masks, and products.

        Args:
            name_image (str): Name of the image.
            path_sen (str): Path to the sentinel data directory.

        Returns:
            tuple[Tensor, Tensor, list[str]]: Sentinel data, masks, and products.
        """
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
    
    def get_sen_idx_temp(self, sen_products) -> Tensor:
        """
        Get sentinel temporal indices.

        Args:
            sen_products (list[str]): List of sentinel products.

        Returns:
            Tensor: Sentinel temporal indices.
        """
        sen_dates = [datetime.strptime(product.split('_')[2], '%Y%m%dT%H%M%S') for product in sen_products]
        sen_channels = [math.ceil(date.month / (12 / self.sen_temp_size)) for date in sen_dates]

        return torch.Tensor(sen_channels) - 1
    
    def sen_filter_cover(self, sen_data: Tensor, sen_masks: Tensor) -> Tensor:
        """
        Filter sentinel data based on cover probability.

        Args:
            sen_data (Tensor): Sentinel data.
            sen_masks (Tensor): Sentinel masks.

        Returns:
            Tensor: Filtered sentinel data.
        """
        cover = (sen_masks >= self.prob_cover).sum(dim=1).to(dtype=torch.bool)
        cover = cover.unsqueeze(1)
        sen_data = sen_data.to(dtype=torch.float32)
        
        return torch.where(cover, sen_data, torch.nan)

    def sen_temporal_reduction(self, sen_data: Tensor, idx_temporal: Tensor) -> Tensor:
        """
        Perform temporal reduction on sentinel data.

        Args:
            sen_data (Tensor): Sentinel data.
            idx_temporal (Tensor): Temporal indices.

        Returns:
            Tensor: Temporally reduced sentinel data.
        """
        sen_shape = list(sen_data.shape)
        sen_shape[0] = self.sen_temp_size
        sen_red = torch.zeros(sen_shape, dtype=torch.float32)
        
        for i_temporal in range(self.sen_temp_size):
            if self.sen_temp_reduc == 'mean':
                sen_red[i_temporal] = torch.nanmean(sen_data[idx_temporal == i_temporal], dim=0)
            elif self.sen_temp_reduc == 'median':
                sen_red[i_temporal] = torch.nanmedian(sen_data[idx_temporal == i_temporal], dim=0)[0]

        return torch.where(torch.isnan(sen_red), 0, sen_red)

    def get_sen(self, name_image: str, path_sen: str, sen_idx_band: list[int]) -> Tensor:
        """
        Get sentinel data for an image.

        Args:
            name_image (str): Name of the image.
            path_sen (str): Path to the sentinel data directory.
            sen_idx_band (list[int]): List of sentinel band indices.

        Returns:
            Tensor: Sentinel data.
        """
        sen_data, sen_masks, sen_products = self.read_sens(name_image, path_sen)
        sen_data = sen_data[:, sen_idx_band]
        sen_data = self.sen_filter_cover(sen_data, sen_masks)
        idx_temporal = self.get_sen_idx_temp(sen_products)
        sen_data = self.sen_temporal_reduction(sen_data, idx_temporal)
        sen_data = sen_data / 10_000
    
        return self.sen_normalize(sen_data)
    
    def get_aerial(self, path_aerial) -> Tensor:
        """
        Get aerial image data.

        Args:
            path_aerial (str): Path to the aerial image.

        Returns:
            Tensor: Aerial image data.
        """
        aerial = tifffile.imread(path_aerial)

        return self.aerial_normalize(aerial)

    def get_labels(self, path_labels) -> Tensor:
        """
        Get labels data.

        Args:
            path_labels (str): Path to the labels data.

        Returns:
            Tensor: Labels data.
        """
        labels = tifffile.imread(path_labels)
        labels = torch.from_numpy(labels)
        labels = labels - 1

        return torch.where(labels < 13, labels, 12)

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.list_images)

    def __getitem__(self, idx) -> tuple[str, Tensor, Tensor, Tensor]:
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple[str, Tensor, Tensor, Tensor]: Image name, aerial data, sentinel data, and labels data.
        """
        path_aerial, path_sen, path_labels, name_image = self.get_paths(idx)
        aerial = self.get_aerial(path_aerial)
        sen = self.get_sen(name_image, path_sen, self.sen_idx_band)

        if self.is_test:
            labels = torch.ByteTensor()
        else:
            labels = self.get_labels(path_labels)

        return name_image, aerial, sen, labels


def get_list_images(path) -> list[str]:
    """
    Get a list of image paths from a directory.

    Args:
        path (str): Path to the directory.

    Returns:
        list[str]: List of image paths.
    """
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
        sen_temp_size=3,
        sen_temp_reduc='median',
        sen_list_bands=['2', '3', '4', '5', '6', '7', '8', '8a', '11', '12'],
        prob_cover=10,
        is_test=False,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
    )

    image_id, aerial, sen, labels = dataset[0]
    print(image_id, aerial.shape, sen.shape, labels.shape)

    for image_id, aerial, sen, labels in dataloader:
        # pass
        print(image_id, aerial.shape, sen.shape, labels.shape)
        break
