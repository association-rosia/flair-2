import torch
import torchvision.transforms.functional as F
from typing import List
import itertools
import math


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


class Perspective(Augmentation):
    def __init__(self, distortion_scale: List):
        distortion_scale = list(set([0] + distortion_scale))
        super().__init__(distortion_scale)
        self.startendpoints_list = []

    @staticmethod
    def get_startpoints(width, height):
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]

        return startpoints

    @staticmethod
    def get_endpoints(width, height, distortion_scale):
        half_height = height // 2
        half_width = width // 2

        top_left = [
            int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1,)).item()),
            int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1,)).item()),
        ]
        top_right = [
            int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1,)).item()),
            int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1,)).item()),
        ]
        bot_right = [
            int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1,)).item()),
            int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1,)).item()),
        ]
        bot_left = [
            int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1,)).item()),
            int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1,)).item()),
        ]

        endpoints = [top_left, top_right, bot_right, bot_left]

        return endpoints

    def augment(self, inputs: dict, distortion_scale=0, **kwargs):
        startendpoints = {}
        is_first = True

        for key in inputs.keys():
            if is_first:
                height = inputs[key].shape[-1]
                width = inputs[key].shape[-2]
                startpoints = self.get_startpoints(width, height)
                endpoints = self.get_endpoints(width, height, distortion_scale)
                startendpoints['startpoints'] = startpoints
                startendpoints['endpoints'] = endpoints
                is_first = False
            else:
                height_previous = height
                height = inputs[key].shape[-1]
                height_rate = height / height_previous

                width_previous = width
                width = inputs[key].shape[-2]
                width_rate = width / width_previous

                startpoints = self.get_startpoints(width, height)
                endpoints = [[math.floor(width_rate * point[-2]), math.floor(height_rate * point[-1])] for point in endpoints]

            inputs[key] = F.perspective(inputs[key], startpoints=startpoints, endpoints=endpoints)

        self.startendpoints_list.append(startendpoints)

        return inputs

    def deaugment(self, output, distortion_scale=0, **kwargs):
        startendpoints = self.startendpoints_list[0]
        self.startendpoints_list.pop(0)

        startpoints = startendpoints['startpoints']
        endpoints = startendpoints['endpoints']

        output = F.perspective(output, startpoints=endpoints, endpoints=startpoints)

        return output
