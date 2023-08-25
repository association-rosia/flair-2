import torch
import torchmetrics

import wandb


class ConfusionMatrix(torchmetrics.Metric):
    """
    Custom metric class for computing and visualizing a confusion matrix using WandB.
    
    This class extends the `torchmetrics.Metric` class to compute and display a confusion matrix
    using the WandB library. It accumulates predictions and target values over multiple batches
    and calculates the confusion matrix when the `compute` method is called.
    
    Args:
        classes (list): List of class names for labeling the confusion matrix.
    
    Attributes:
        target (list): List to store target values.
        preds (list): List to store predicted values.
        classes (list): List of class names for labeling the confusion matrix.
    
    Methods:
        update(self, preds, target):
            Accumulates predictions and target values for computing the confusion matrix.
        
        compute(self):
            Computes the confusion matrix using accumulated values and returns a WandB plot.
        
        reset(self):
            Resets the accumulated target and predicted values.
    """
    def __init__(self, classes) -> None:
        super().__init__()

        self.target = []
        self.preds = []
        self.classes = classes

    def update(self, preds, target):
        self.target.append(target)
        self.preds.append(preds)

    def compute(self):
        target = torch.cat(self.target, dim=0)
        target = torch.flatten(target)
        target = target.tolist()

        preds = torch.cat(self.preds, dim=0)
        preds = torch.argmax(preds, dim=1)
        preds = torch.flatten(preds)
        preds = preds.tolist()

        return wandb.plot.confusion_matrix(preds=preds, y_true=target, class_names=self.classes,
                                           title='Confusion matrix by pixel')

    def reset(self):
        self.target = []
        self.preds = []
