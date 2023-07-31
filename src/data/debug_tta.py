import tta
from torch import nn


class FakeModel(nn.Module):

    def forward(self, sen, aerial):
        return sen


augmentations = tta.augmentations.Augmentations([
    tta.augmentations.HorizontalFlip(),
    tta.augmentations.VerticalFlip(),
    tta.augmentations.Rotate([90, 180, 270]),
    tta.augmentations.Solarize([0, 0.25, 0.5, 0.75])
])

wrapper = tta.wrappers.SegmentationWrapper()
