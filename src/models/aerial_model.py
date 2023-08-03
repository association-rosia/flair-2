from torch import nn
import segmentation_models_pytorch as smp

class AerialModel(nn.Module):
    def __init__(self, architecture, encoder_name, encoder_weight, num_classes) -> None:
        super().__init__()
        
        self.model = getattr(smp, architecture)(
            encoder_name=encoder_name,
            encoder_weights=encoder_weight,
            in_channels=5,
            activation="softmax",
            classes=num_classes,
        )
        
    def forward(self, aerial, *args, **kwargs):
        return self.model(aerial)