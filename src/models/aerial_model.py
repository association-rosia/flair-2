import segmentation_models_pytorch as smp
from torch import nn


class AerialModel(nn.Module):
    """
    AerialModel class for creating a segmentation model for aerial imagery.
    """
    def __init__(self, arch, encoder_name, num_classes) -> None:
        """
        Initialize the AerialModel.

        Args:
            arch (str): Architecture of the model.
            encoder_name (str): Name of the encoder architecture.
            num_classes (int): Number of segmentation classes.
        """
        super().__init__()

        # Create the segmentation model
        self.model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            classes=num_classes,
            in_channels=5,
        )

    def forward(self, aerial, *args, **kwargs):
        """
        Forward pass through the AerialModel.

        Args:
            aerial (tensor): Input aerial imagery tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            tensor: Model's output tensor.
        """
        return self.model(aerial)
