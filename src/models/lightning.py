import os
import shutil
from time import time
import json

import numpy as np
import pytorch_lightning as pl
import tifffile as tiff
import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassJaccardIndex

import src.data.tta.augmentations as agms
import src.data.tta.wrappers as wrps
from src.constants import get_constants
from src.data.make_dataset import FLAIR2Dataset
from src.models.aerial_model import AerialModel
from src.models.multimodal_model import MultiModalSegformer

cst = get_constants()


class FLAIR2Lightning(pl.LightningModule):
    """
    Lightning Module for the FLAIR-2 project.
    """

    def __init__(
            self,
            arch_lib,
            arch,
            encoder_name,
            classes,
            learning_rate,
            class_weights,
            list_images_train,
            list_images_val,
            list_images_test,
            aerial_list_bands,
            sen_size,
            sen_temp_size,
            sen_temp_reduc,
            sen_list_bands,
            prob_cover,
            use_augmentation,
            use_tta,
            train_batch_size,
            test_batch_size
    ):
        super(FLAIR2Lightning, self).__init__()
        self.step = None
        self.save_hyperparameters(logger=False)

        # Initialize hyperparameters and configurations
        self.arch_lib = arch_lib
        self.arch = arch
        self.encoder_name = encoder_name
        self.classes = classes
        self.class_labels = {key: label for key, label in enumerate(self.classes)}
        self.num_classes = len(classes)
        self.learning_rate = learning_rate
        self.class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self.list_images_train = list_images_train
        self.list_images_val = list_images_val
        self.list_images_test = list_images_test
        self.aerial_list_bands = aerial_list_bands
        self.sen_size = sen_size
        self.sen_temp_size = sen_temp_size
        self.sen_temp_reduc = sen_temp_reduc
        self.sen_list_bands = sen_list_bands
        self.prob_cover = prob_cover
        self.use_augmentation = use_augmentation
        self.use_tta = use_tta
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.tta_limit = 1  # init TTA to mim value possible
        self.path_predictions = None
        self.log_image_idx = None

        if self.arch_lib == 'custom':
            self.model = MultiModalSegformer.from_pretrained(
                pretrained_model_name_or_path=self.arch,
                num_labels=self.num_classes,
                num_channels=len(self.aerial_list_bands),
                ignore_mismatched_sizes=True
            )
        else:
            # Create the AerialModel
            self.model = AerialModel(
                arch_lib=self.arch_lib,
                arch=self.arch,
                encoder_name=self.encoder_name,
                num_channels=len(self.aerial_list_bands),
                num_classes=self.num_classes
            )

        if self.use_tta:
            # Initialize augmentations for tta
            augmentations = agms.Augmentations([
                agms.HorizontalFlip(),
                agms.VerticalFlip(),
                agms.Rotate([90, 180, 270]),
                # not a good idea to use this transformation for multi-class segmentation
                # agms.Perspective([0.25, 0.375, 0.5])
            ])
            self.model = wrps.SegmentationWrapper(model=self.model, augmentations=augmentations)

        # init metrics for evaluation
        self.metrics = MetricCollection({
                'val/miou': MulticlassJaccardIndex(self.num_classes, average='macro')
            })
        
        self.inverse_normalize = self.init_inverse_normalize(self.aerial_list_bands)

    @staticmethod
    def init_inverse_normalize(aerial_list_bands):
        path_aerial_pixels_metadata = os.path.join(cst.path_data, 'aerial_pixels_metadata.json')
        with open(path_aerial_pixels_metadata) as f:
            stats = json.load(f)
        
        aerial_idx_band = [cst.aerial_band2idx[str(band)] for band in aerial_list_bands]
        
        mean = -torch.Tensor(stats['mean']) / torch.Tensor(stats['std'])
        std = 1 / torch.Tensor(stats['std'])
        
        return T.Normalize(
            mean=mean[aerial_idx_band],
            std=std[aerial_idx_band]
        )

    def forward(self, inputs):
        if self.use_tta:
            x = self.model(inputs=inputs, step=self.step, batch_size=self.train_batch_size, limit=self.tta_limit)
        else:
            x = self.model(**inputs)
        return x

    def on_train_epoch_start(self) -> None:
        self.step = 'training'

    def training_step(self, batch):
        _, aerial, sen, labels = batch
        outputs = self.forward(inputs={'aerial': aerial, 'sen': sen})
        labels = labels.to(dtype=torch.int64)
        loss = self.criterion(outputs, labels)
        self.log('train/loss', loss, on_step=True, on_epoch=True)

        return loss

    def on_validation_epoch_start(self) -> None:
        self.step = 'validation'

    def log_aerial_mask(self, aerial, mask_target, mask_pred):
        image = self.inverse_normalize(aerial)
        image = image[:3]
        image = image.permute(1, 2, 0)
        image = image.numpy(force=True)
        image = image * 255.0
        image = image.astype(np.uint8)

        mask_pred = mask_pred.softmax(dim=0)
        mask_pred = mask_pred.argmax(dim=0)
        mask_pred = mask_pred.numpy(force=True)
        mask_pred = mask_pred.astype(np.uint8)

        mask_target = mask_target.numpy(force=True)
        mask_target = mask_target.astype(np.uint8)

        self.logger.experiment.log(
            {'aerial_image': wandb.Image(
                image,
                masks={
                    'predictions': {
                        'mask_data': mask_pred,
                        'class_labels': self.class_labels
                    },
                    'ground_truth': {
                        'mask_data': mask_target,
                        'class_labels': self.class_labels
                    }})})

    def validation_step(self, batch, batch_idx):
        _, aerial, sen, labels = batch
        outputs = self.forward(inputs={'aerial': aerial, 'sen': sen})

        labels = labels.to(dtype=torch.int64)
        loss = self.criterion(outputs, labels)

        self.log('val/loss', loss, on_step=True, on_epoch=True)
        self.metrics.update(outputs, labels)

        if batch_idx == self.log_image_idx // self.train_batch_size:
            batch_image_idx = self.log_image_idx % self.train_batch_size
            self.log_aerial_mask(aerial[batch_image_idx], labels[batch_image_idx], outputs[batch_image_idx])

        return loss

    def on_validation_epoch_end(self) -> None:
        # Compute metrics
        metrics = self.metrics.compute()

        # Log metrics
        self.log_dict(metrics, on_epoch=True)

        # Reset metrics
        self.metrics.reset()

        return metrics

    def on_test_epoch_start(self) -> None:
        self.step = 'test'

    def test_step(self, batch, batch_idx):
        image_ids, aerial, sen, _ = batch

        outputs = self.forward(inputs={'aerial': aerial, 'sen': sen})
        outputs = outputs.softmax(dim=1)
        outputs = outputs.argmax(dim=1)

        # for pred_label, img_id in zip(outputs, image_ids):
        #     img = pred_label.numpy(force=True)
        #     img = img.astype(dtype=np.uint8)
        #     img_path = os.path.join(self.path_predictions, f'PRED_{img_id}')
        #     tiff.imwrite(img_path, img, dtype=np.uint8, compression='LZW')

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)

        # scheduler = ReduceLROnPlateau(
        #     optimizer=optimizer,
        #     mode='min',
        #     factor=0.5,
        #     patience=5,
        #     verbose=True
        # )

        # return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val/loss'}}
        return optimizer

    def train_dataloader(self):
        # Initialize training dataset and data loader
        dataset_train = FLAIR2Dataset(
            list_images=self.list_images_train,
            aerial_list_bands=self.aerial_list_bands,
            sen_size=self.sen_size,
            sen_temp_size=self.sen_temp_size,
            sen_temp_reduc=self.sen_temp_reduc,
            sen_list_bands=self.sen_list_bands,
            prob_cover=self.prob_cover,
            use_augmentation=self.use_augmentation,
            use_tta=self.use_tta,
            is_val=False,
            is_test=False,
        )

        return DataLoader(
            dataset=dataset_train,
            batch_size=self.train_batch_size,
            num_workers=cst.train_num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        # Initialize validation dataset and data loader
        dataset_val = FLAIR2Dataset(
            list_images=self.list_images_val,
            aerial_list_bands=self.aerial_list_bands,
            sen_size=self.sen_size,
            sen_temp_size=self.sen_temp_size,
            sen_temp_reduc=self.sen_temp_reduc,
            sen_list_bands=self.sen_list_bands,
            prob_cover=self.prob_cover,
            use_augmentation=self.use_augmentation,
            use_tta=self.use_tta,
            is_val=True,
            is_test=False,
        )

        return DataLoader(
            dataset=dataset_val,
            batch_size=self.train_batch_size,
            num_workers=cst.train_num_workers,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self):
        # Initialize test dataset and data loader
        dataset_test = FLAIR2Dataset(
            list_images=self.list_images_test,
            aerial_list_bands=self.aerial_list_bands,
            sen_size=self.sen_size,
            sen_temp_size=self.sen_temp_size,
            sen_temp_reduc=self.sen_temp_reduc,
            sen_list_bands=self.sen_list_bands,
            prob_cover=self.prob_cover,
            use_augmentation=self.use_augmentation,
            use_tta=self.use_tta,
            is_val=False,
            is_test=True,
        )

        return DataLoader(
            dataset=dataset_test,
            batch_size=self.test_batch_size,
            num_workers=cst.test_num_workers,
            shuffle=False,
            drop_last=False,
        )
