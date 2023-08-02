import torchvision.transforms.functional as F
from typing import List
import itertools


class Augmentation:
    def __init__(self, params):
        super().__init__()
        self.params = params

    def apply(self, inputs: dict, *args, **params):
        raise NotImplementedError

    def de_apply(self, output, *args, **params):
        raise NotImplementedError


class Augmentations:
    def __init__(self, augmentations: List[Augmentation]):
        self.list = augmentations
        self.params = [t.params for t in self.list]
        self.product = list(itertools.product(*self.params))
        self.de_list = self.list[::-1]
        self.de_product = [p[::-1] for p in self.product]


class HorizontalFlip(Augmentation):
    def __init__(self):
        super().__init__([True, False])  # apply

    def apply(self, inputs: dict, apply=False, **kwargs):
        if apply:
            for key in inputs.keys():
                inputs[key] = F.hflip(inputs[key])

        return inputs

    def de_apply(self, output, apply=False, **kwargs):
        if apply:
            output = F.hflip(output)

        return output


class VerticalFlip(Augmentation):
    def __init__(self):
        super().__init__([True, False])  # apply

    def apply(self, inputs: dict, apply=False, **kwargs):
        if apply:
            for key in inputs.keys():
                inputs[key] = F.vflip(inputs[key])

        return inputs

    def de_apply(self, output, apply=False, **kwargs):
        if apply:
            output = F.vflip(output)

        return output


class Rotate(Augmentation):
    def __init__(self, angles):
        super().__init__(angles)
        self.allowed_angles = [0, 90, 180, 270]
        self.angles = list(set([0] + angles))  # angle = 0 is not change

        for angle in angles:
            if angle not in self.allowed_angles:
                raise ValueError(f'angles must be equal to 0, 90, 180 or 270')

    def apply(self, inputs: dict, angle=0, **kwargs):
        for key in inputs.keys():
            inputs[key] = F.rotate(inputs[key], angle=angle)

        return inputs

    def de_apply(self, output, angle=0, **kwargs):
        output = F.rotate(output, angle=-angle)

        return output


class Solarize(Augmentation):
    def __init__(self, thresholds):
        super().__init__(thresholds)
        self.thresholds = list(set(thresholds + [1]))  # threshold = 1 is not change

    def apply(self, inputs: dict, threshold=1, **kwargs):
        for key in inputs.keys():
            inputs[key] = F.solarize(inputs[key], threshold=threshold)

        return inputs

    def de_apply(self, output: dict, threshold=1, **kwargs):
        # this transformation do not "destruct" the inputs
        return output
