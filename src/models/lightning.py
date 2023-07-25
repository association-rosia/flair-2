import cv2

import torch
import torch.nn as nn
from torch.optim import AdamW
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics import MetricCollection
from src.models.metrics import ConfusionMatrix

import segmentation_models_pytorch as smp


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

        self.model = getattr(smp, self.architecture)(
            encoder_name=encoder_name,
            encoder_weights=encoder_weight,
            in_channels=5,
            activation="softmax",
            classes=self.num_classes
        )
        
        self.metrics = MetricCollection({
            "val/MIoU": MulticlassJaccardIndex(self.num_classes, average="macro"),
            "val/IoU": MulticlassJaccardIndex(self.num_classes, average="none"),
            "val/confusion_matrix": ConfusionMatrix(self.classes),
        })

    def forward(self, inputs):
        x = self.model(inputs)

        return x

    def training_step(self, batch, batch_idx):
        aerial, sen, labels = batch
        outputs = self.forward(aerial)
        labels = torch.squeeze(labels).to(dtype=torch.int64)
        loss = self.criterion(outputs, labels)
        self.log('train/loss', loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        aerial, sen, labels = batch
        outputs = self.forward(aerial)
        
        # BUG: Log image on wandb
        # if batch_idx == 0:
        #     prediction = torch.argmax(outputs[0], dim=2)
        #     self.logger.log_image(key="val/prediction", images=[prediction])
        labels = torch.squeeze(labels).to(dtype=torch.int64)
        loss = self.criterion(outputs, labels)
        
        self.log('val/loss', loss, on_epoch=True)
        self.metrics.update(outputs, labels)
        
        return loss

    def on_validation_epoch_end(self) -> None:
        # Compute metrics
        metrics = self.metrics.compute()
        
        # Split 1d array tensor IoU metrics to value tensor
        # Add class name and associate value into the formatted metrics dict
        formatted_metrics = {}
        for key, value in metrics.items():
            # Metrics that return 1d array tensor 
            spe_key = "val/IoU"
            if key == spe_key:
                for i, class_name in enumerate(self.classes):
                    formatted_metrics[spe_key + "-" + class_name] = value[i]
                continue
            formatted_metrics[key] = value
        
        # Confusion matrix need a special method to be logged
        self.logger.experiment.log(formatted_metrics)
        # Reset metrics
        self.metrics.reset()

        return metrics

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)

        return optimizer
