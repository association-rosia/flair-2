import segmentation_models_pytorch as smp
from torch import nn
from transformers import SegformerForSemanticSegmentation


class AerialModel(nn.Module):
    """
    AerialModel class for creating a segmentation model for aerial imagery.
    """

    def __init__(self, arch_lib, arch, encoder_name, num_classes) -> None:
        """
        Initialize the AerialModel.

        Args:
            arch_lib (str): SMP or Transformers
            arch (str): Architecture of the model.
            encoder_name (str): Name of the encoder architecture.
            num_classes (int): Number of segmentation classes.
        """
        super().__init__()
        self.arch_lib = arch_lib
        self.arch = arch
        self.encoder_name = encoder_name
        self.num_classes = num_classes

        if self.arch_lib == 'SMP':
            # Create the segmentation model using SMP
            self.model = smp.create_model(
                arch=self.arch,
                encoder_name=self.encoder_name,
                classes=self.num_classes,
                in_channels=5,
            )
        elif self.arch_lib == 'HF':
            # Create the segmentation model using Transformers
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_model_name_or_path=self.arch,
                num_labels=self.num_classes,
                num_channels=5,
                ignore_mismatched_sizes=True
            )
        else:
            raise ValueError('arch_lib must SMP or HF')

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
        if self.arch_lib == 'SMP':
            # Infer the segmentation model using SMP
            outputs = self.model(aerial)
        elif self.arch_lib == 'HF':
            # Infer the segmentation model using Transformers
            outputs = self.model(pixel_values=aerial)
            outputs = nn.functional.interpolate(
                input=outputs.logits,
                size=aerial.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        else:
            raise ValueError('arch_lib must SMP or HF (Hugging Face)')

        return outputs
