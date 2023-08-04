import torch
import torchmetrics

import wandb


class ConfusionMatrix(torchmetrics.Metric):
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
        
        return wandb.plot.confusion_matrix(preds=preds, y_true=target, class_names=self.classes, title='Confusion matrix by pixel')
        
        
    def reset(self):
        self.target = []
        self.preds = []
        