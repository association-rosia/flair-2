import os

import numpy as np
import pytorch_lightning as pl
import tifffile as tiff
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassJaccardIndex

import src.data.tta.augmentations as agms
import src.data.tta.wrappers as wrps

from src.constants import get_constants
from src.data.make_dataset import FLAIR2Dataset
from src.models.aerial_model import AerialModel

cst = get_constants()


class FLAIR2Lightning(pl.LightningModule):
    """
    Lightning Module for the FLAIR-2 project.
    """
    def __init__(
            self,
            arch,
            encoder_name,
            classes,
            learning_rate,
            criterion_weight,
            list_images_train,
            list_images_val,
            list_images_test,
            sen_size,
            use_augmentation,
            batch_size,
    ):
        super(FLAIR2Lightning, self).__init__()
        self.save_hyperparameters()

        # Initialize hyperparameters and configurations
        self.arch = arch
        self.encoder_name = encoder_name
        self.classes = classes
        self.num_classes = len(classes)
        self.learning_rate = learning_rate
        self.criterion_weight = torch.as_tensor(criterion_weight, dtype=torch.float32)
        self.criterion = nn.CrossEntropyLoss(weight=self.criterion_weight)
        self.list_images_train = list_images_train
        self.list_images_val = list_images_val
        self.list_images_test = list_images_test
        self.sen_size = sen_size
        self.use_augmentation = use_augmentation
        self.batch_size = batch_size
        self.path_predictions = None

        # Create the AerialModel
        self.model = AerialModel(
            arch=self.arch,
            encoder_name=self.encoder_name,
            num_classes=self.num_classes
        )

        # Initialize augmentations
        augmentations = agms.Augmentations([
            agms.HorizontalFlip(),
            agms.VerticalFlip(),
            agms.Rotate([90, 180, 270]),
            agms.Perspective([0.25, 0.5, 0.75])
        ])

        if use_augmentation:
            self.model = wrps.SegmentationWrapper(model=self.model, augmentations=augmentations)

        # Initialize metrics for evaluation
        self.metrics = MetricCollection(
            {
                'MIoU': MulticlassJaccardIndex(self.num_classes, average='macro')
            }
        )

    def forward(self, inputs):
        if self.use_augmentation:
            x = self.model(inputs=inputs, step=self.step, batch_size=self.batch_size, limit=10)
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

    def validation_step(self, batch, batch_idx):
        _, aerial, sen, labels = batch
        outputs = self.forward(inputs={'aerial': aerial, 'sen': sen})

        labels = labels.to(dtype=torch.int64)
        loss = self.criterion(outputs, labels)

        self.log('val/loss', loss, on_epoch=True)
        self.metrics.update(outputs, labels)

        return loss

    def on_validation_epoch_end(self) -> None:
        # Compute metrics
        metrics = self.metrics.compute()

        self.logger.experiment.log(metrics)
        # Reset metrics
        self.metrics.reset()

        return metrics

    def on_test_epoch_start(self) -> None:
        self.step = 'test'

    def test_step(self, batch, batch_idx):
        image_ids, aerial, sen, _ = batch

        outputs = self.forward(inputs={'aerial': aerial, 'sen': sen})
        pred_labels = torch.argmax(outputs, dim=1)

        # * Challenge rule: set the data type of the image files as Byte (uint8)
        # * with values ranging from 0 to 12

        # ! Do not uncomment the folowing line, read the comment above.
        # pred_labels += 1

        for pred_label, img_id in zip(pred_labels, image_ids):
            img = pred_label.numpy(force=True)
            img = img.astype(np.uint8)
            img_path = os.path.join(self.path_predictions, f'PRED_{img_id}')
            tiff.imwrite(img_path, img, dtype=np.uint8, compression='LZW')

        return pred_labels

    def on_predict_epoch_start(self) -> None:
        self.step = 'predict'

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        # Initialize training dataset and data loader
        dataset_train = FLAIR2Dataset(
            list_images=self.list_images_train,
            sen_size=self.sen_size,
            is_test=False,
        )

        return DataLoader(
            dataset=dataset_train,
            batch_size=self.batch_size,
            num_workers=cst.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        # Initialize validation dataset and data loader
        dataset_val = FLAIR2Dataset(
            list_images=self.list_images_val,
            sen_size=self.sen_size,
            is_test=False,
        )

        return DataLoader(
            dataset=dataset_val,
            batch_size=self.batch_size,
            num_workers=cst.num_workers,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self):
        # Initialize test dataset and data loader
        dataset_test = FLAIR2Dataset(
            list_images=self.list_images_test,
            sen_size=self.sen_size,
            is_test=True,
        )

        return DataLoader(
            dataset=dataset_test,
            batch_size=self.batch_size,
            num_workers=cst.num_workers,
            shuffle=False,
            drop_last=False,
        )
