import torchvision.transforms.functional as F
from typing import List
import itertools


class Augmentation:
    def __init__(self, params):
        super().__init__()
        self.params = params

    def apply(self, inputs, *args, **params):  # TODO: change inputs to {'input1': value, 'input2': value, ...}
        raise NotImplementedError

    def de_apply(self, output, *args, **params):  # TODO: change output to {'output': value}
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

    def apply(self, inputs, apply=False, **kwargs):  # TODO: change inputs to {'input1': value, 'input2': value, ...}
        augmented_inputs = []
        for input in inputs:
            input = F.hflip(input)
            augmented_inputs.append(input)

        return augmented_inputs

    def de_apply(self, output, apply=False, **kwargs):  # TODO: change output to {'output': value}
        output = F.hflip(output)

        return output


class VerticalFlip(Augmentation):
    def __init__(self):
        super().__init__([True, False])  # apply

    def apply(self, inputs, apply=False, **kwargs):  # TODO: change inputs to {'input1': value, 'input2': value, ...}
        augmented_inputs = []
        for input in inputs:
            input = F.vflip(input)
            augmented_inputs.append(input)

        return augmented_inputs

    def de_apply(self, output, apply=False, **kwargs):  # TODO: change output to {'output': value}
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

    def apply(self, inputs, angle=0, **kwargs):  # TODO: change inputs to {'input1': value, 'input2': value, ...}
        augmented_inputs = []
        for input in inputs:
            input = F.rotate(input, angle=angle)
            augmented_inputs.append(input)

        return augmented_inputs

    def de_apply(self, output, angle=0, **kwargs):  # TODO: change output to {'output': value}
        output = F.rotate(input, angle=-angle)

        return output


class Solarize(Augmentation):
    def __init__(self, thresholds):
        super().__init__(thresholds)
        self.thresholds = list(set(thresholds + [1]))  # threshold = 1 is not change

    def apply(self, inputs, threshold=1, **kwargs):  # TODO: change inputs to {'input1': value, 'input2': value, ...}
        augmented_inputs = []
        for input in inputs:
            input = F.solarize(input, threshold=threshold)
            augmented_inputs.append(input)

        return augmented_inputs

    def de_apply(self, output, threshold=1, **kwargs):  # TODO: change output to {'output': value}
        # this transformation do not "destruct" the inputs
        return output
