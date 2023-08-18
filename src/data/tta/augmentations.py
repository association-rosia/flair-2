import torchvision.transforms.functional as F
from typing import List
import itertools


class Augmentation:
    def __init__(self, params):
        self.params = params
        # TODO: add identity param value to reconstruct be sure to get the original inputs even if we are using the
        #  limit parameter
        super().__init__()

    def augment(self, inputs: dict, *args, **params):
        raise NotImplementedError

    def deaugment(self, output, *args, **params):
        raise NotImplementedError


class Augmentations:
    def __init__(self, augmentations: List[Augmentation]):
        self.list = augmentations
        self.params = [t.params for t in self.list]
        self.product = list(itertools.product(*self.params))
        self.delist = self.list[::-1]


class HorizontalFlip(Augmentation):
    def __init__(self):
        super().__init__([True, False])  # apply

    def augment(self, inputs: dict, apply=False, **kwargs):
        if apply:
            for key in inputs.keys():
                inputs[key] = F.hflip(inputs[key])

        return inputs

    def deaugment(self, output, apply=False, **kwargs):
        if apply:
            output = F.hflip(output)

        return output


class VerticalFlip(Augmentation):
    def __init__(self):
        super().__init__([True, False])  # apply

    def augment(self, inputs: dict, apply=False, **kwargs):
        if apply:
            for key in inputs.keys():
                inputs[key] = F.vflip(inputs[key])

        return inputs

    def deaugment(self, output, apply=False, **kwargs):
        if apply:
            output = F.vflip(output)

        return output


class Rotate(Augmentation):
    def __init__(self, angles: List):
        allowed_angles = [0, 90, 180, 270]
        angles = list(set([0] + angles))  # angle = 0 is not change

        for angle in angles:
            if angle not in allowed_angles:
                raise ValueError(f'angles must be equal to 0, 90, 180 or 270')

        super().__init__(angles)

    def augment(self, inputs: dict, angle=0, **kwargs):
        for key in inputs.keys():
            inputs[key] = F.rotate(inputs[key], angle=angle)

        return inputs

    def deaugment(self, output, angle=0, **kwargs):
        output = F.rotate(output, angle=-angle)

        return output


# class Solarize(Augmentation):
#     def __init__(self, thresholds: List):
#         self.thresholds = thresholds
#         thresholds = list(set([1] + thresholds))  # threshold = 1 is not change
#
#         for threshold in thresholds:
#             if threshold < 0 or threshold > 1:
#                 raise ValueError(f'thresholds must be between 0 and 1')
#
#         super().__init__(thresholds)
#
#     def augment(self, inputs: dict, threshold=1, **kwargs):
#         for key in inputs.keys():
#             inputs[key] = F.solarize(inputs[key], threshold=threshold)
#
#         return inputs
#
#     def deaugment(self, output, threshold=1, **kwargs):
#         return output

class Augmentation:
    def __init__(self, params):
        self.params = params
        super().__init__()

    def augment(self, inputs: dict, *args, **params):
        raise NotImplementedError

    def deaugment(self, output, *args, **params):
        raise NotImplementedError
