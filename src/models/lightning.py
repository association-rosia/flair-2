import os

import numpy as np

import torch
import torch.nn as nn
from torch.optim import AdamW
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics import MetricCollection
from src.models.metrics import ConfusionMatrix

import segmentation_models_pytorch as smp
import ttach as tta

import tifffile as tiff

from src.constants import get_constants

cst = get_constants()


class FLAIR2Lightning(pl.LightningModule):
    def __init__(self, architecture, encoder_name, encoder_weight, classes, learning_rate, criterion_weight):
        super(FLAIR2Lightning, self).__init__()
        self.save_hyperparameters()

        self.classes = classes
        self.num_classes = len(classes)
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.criterion_weight = torch.as_tensor(criterion_weight, dtype=torch.float32)
        self.criterion = nn.CrossEntropyLoss(weight=self.criterion_weight)
        self.apply_tta = None
        self.path_submissions = None

        self.model = getattr(smp, self.architecture)(
            encoder_name=encoder_name,
            encoder_weights=encoder_weight,
            in_channels=5,
            activation="softmax",
            classes=self.num_classes
        )
        
        self.metrics = MetricCollection({
            "MIoU": MulticlassJaccardIndex(self.num_classes, average="macro"),
            "IoU": MulticlassJaccardIndex(self.num_classes, average="none"),
            "confusion_matrix": ConfusionMatrix(self.classes),
        })
        
        self.tta_transforms = tta.Compose(
            [
                # ! Comment next line if classes include Left / Right position
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                # tta.FiveCrops(crop_height=5, crop_width=5),
                tta.Rotate90([0, 90, 180, 270])
            ]
        )

    def forward(self, inputs):
        x = self.model(inputs)

        return x

    def training_step(self, batch):
        aerial, sen, labels = batch
        outputs = self.forward(aerial)
        labels = torch.squeeze(labels).to(dtype=torch.int64)
        loss = self.criterion(outputs, labels)
        self.log('train/loss', loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        aerial, sen, labels = batch
        outputs = self.forward(aerial)
        
        labels = torch.squeeze(labels).to(dtype=torch.int64)
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
    
    # def log_metrics(self, metrics):
    #     formatted_metrics = {}
    #     for key, value in metrics.items():
    #         # Metrics that return 1d array tensor 
    #         # spe_key = "/IoU"
    #         # if spe_key in key:
    #         #     for i, class_name in enumerate(self.classes):
    #         #         formatted_metrics[spe_key + "-" + class_name] = value[i]
    #         #     continue
    #         formatted_metrics[key] = value
        
    #     # Confusion matrix need a special method to be logged
    #     self.logger.experiment.log(formatted_metrics)

    def enable_tta(self):
        # Wrappe the model to use tta
        self.model = tta.SegmentationTTAWrapper(
            self.model,
            merge_mode='mean',
            transforms=self.tta_transforms
        )
    
    def predict_step(self, batch):
        image_ids, aerial, sen, _ = batch
        
        outputs = self.forward(aerial)
        pred_labels = torch.argmax(outputs, dim=1)
        
        # * Challenge rule: set the data type of the image files as Byte (uint8)
        # * with values ranging from 0 to 12
        
        # ! Do not uncomment the folowing line, read the comment above.
        # pred_labels += 1
        
        for pred_label, img_id in zip(pred_labels, image_ids):
            img: np.ndarray = pred_label.numpy(force=True)
            img = img.astype(np.uint8)
            img_path = os.path.join(self.path_submissions, f'PRED_{img_id}.tif')
            tiff.imwrite(img_path, img)
        
        return pred_labels

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)

        return optimizer
