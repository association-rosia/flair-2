import os
import json

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryJaccardIndex

import src.data.tta.augmentations as agms
import src.data.tta.wrappers as wrps
from src.constants import get_constants
from src.data.make_dataset import FLAIR2Dataset
from src.models.aerial_model import AerialModel
from src.models.multimodal_model import MultiModalSegformer
from src.models.config_model import FLAIR2ConfigModel

cst = get_constants()


class FLAIR2LightningOneVsAll(pl.LightningModule):
    """
    Lightning Module for the FLAIR-2 project.
    """

    def __init__(
            self,
            config: FLAIR2ConfigModel,
    ):
        super(FLAIR2LightningOneVsAll, self).__init__()
        self.step = None
        self.save_hyperparameters(logger=False)

        # Initialize hyperparameters and configurations
        self.config = config
        self.class_labels = {key: label for key, label in enumerate(self.config.classes)}
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([self.config.pos_weight]))
        self.tta_limit = 1  # init TTA to mim value possible
        self.path_predictions = None
        self.log_image_idx = None
        num_classes = 1 if self.config.one_vs_all else len(self.config.classes)

        if self.config.arch_lib == 'custom':
            self.model = MultiModalSegformer.from_pretrained(
                pretrained_model_name_or_path=self.config.arch,
                num_labels=num_classes,
                num_channels=len(self.config.aerial_list_bands),
                ignore_mismatched_sizes=True
            )
        else:
            # Create the AerialModel
            self.model = AerialModel(
                arch_lib=self.config.arch_lib,
                arch=self.config.arch,
                encoder_name=self.config.encoder_name,
                num_channels=len(self.config.aerial_list_bands),
                num_classes=num_classes
            )

        if self.config.use_tta:
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
                'val/miou': BinaryJaccardIndex()
            })
        
        self.inverse_normalize = self.init_inverse_normalize(self.config.aerial_list_bands)

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
        if self.config.use_tta:
            x = self.model(inputs=inputs, step=self.step, batch_size=self.config.train_batch_size, limit=self.tta_limit)
        else:
            x = self.model(**inputs)
        return x.squeeze(dim=1)

    def on_train_epoch_start(self) -> None:
        self.step = 'training'

    def training_step(self, batch):
        _, aerial, sen, labels = batch
        outputs = self.forward(inputs={'aerial': aerial, 'sen': sen})
        labels = labels.float()
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

        mask_pred = torch.where(mask_pred > 0.5, 1, 0)
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

        labels = labels.float()
        loss = self.criterion(outputs, labels)

        self.log('val/loss', loss, on_step=True, on_epoch=True)
        self.metrics.update(outputs, labels)

        if batch_idx == self.log_image_idx // self.config.train_batch_size:
            batch_image_idx = self.log_image_idx % self.config.train_batch_size
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

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config.learning_rate)

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
            list_images=self.config.list_images_train,
            aerial_list_bands=self.config.aerial_list_bands,
            sen_size=self.config.sen_size,
            sen_temp_size=self.config.sen_temp_size,
            sen_temp_reduc=self.config.sen_temp_reduc,
            sen_list_bands=self.config.sen_list_bands,
            prob_cover=self.config.prob_cover,
            use_augmentation=self.config.use_augmentation,
            use_tta=self.config.use_tta,
            one_vs_all=self.config.one_vs_all,
            is_val=False,
            is_test=False,
        )

        return DataLoader(
            dataset=dataset_train,
            batch_size=self.config.train_batch_size,
            num_workers=cst.train_num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        # Initialize validation dataset and data loader
        dataset_val = FLAIR2Dataset(
            list_images=self.config.list_images_val,
            aerial_list_bands=self.config.aerial_list_bands,
            sen_size=self.config.sen_size,
            sen_temp_size=self.config.sen_temp_size,
            sen_temp_reduc=self.config.sen_temp_reduc,
            sen_list_bands=self.config.sen_list_bands,
            prob_cover=self.config.prob_cover,
            use_augmentation=self.config.use_augmentation,
            use_tta=self.config.use_tta,
            one_vs_all=self.config.one_vs_all,
            is_val=True,
            is_test=False,
        )

        return DataLoader(
            dataset=dataset_val,
            batch_size=self.config.train_batch_size,
            num_workers=cst.train_num_workers,
            shuffle=False,
            drop_last=True,
        )