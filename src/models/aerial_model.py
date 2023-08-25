import segmentation_models_pytorch as smp
from torch import nn


class AerialModel(nn.Module):
    def __init__(self, arch, encoder_name, num_classes) -> None:
        super().__init__()

        self.model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            classes=num_classes,
            in_channels=5,
        )

    def forward(self, aerial, *args, **kwargs):
        return self.model(aerial)
