import torch
from torch import nn
import random
import matplotlib.pyplot as plt


class SegmentationWrapper(nn.Module):
    def __init__(self, model, augmentations):
        super().__init__()
        self.model = model
        self.product = augmentations.product
        self.list = augmentations.list
        self.de_product = augmentations.de_product
        self.de_list = augmentations.de_list
        # self.aggregation = aggregation

    def forward(self, inputs, step, limit=None):
        # step can be 'train', 'val, 'test'

        if step == 'train':
            params = random.choice(self.product)

            for augmentation, param in zip(self.list, params):
                inputs = augmentation.apply(inputs, param)

            output = self.model(**inputs)

            de_params = params[::-1]
            for de_augmentation, de_param in zip(self.de_list, de_params):
                output = de_augmentation.de_apply(output, de_param)

        elif step == 'val' or step == 'test':
            product = self.product if limit is None else random.choices(self.product, k=limit)
            de_product = [p[::-1] for p in self.product]

            tta_inputs = {key: [] for key in inputs.keys()}
            for params in product:

                augmented_inputs = inputs.copy()

                for augmentation, param in zip(self.list, params):
                    augmented_inputs = augmentation.apply(augmented_inputs, param)

                for key in inputs.keys():
                    tta_inputs[key].append(augmented_inputs[key])

            for key in tta_inputs.keys():
                tta_inputs[key] = torch.stack(tta_inputs[key], dim=0)

            tta_output = self.model(**tta_inputs)

            for i, de_params in enumerate(de_product):

                for de_augmentation, de_param in zip(self.de_list, de_params):
                    tta_output[i] = de_augmentation.de_apply(tta_output[i], de_param)

            output = torch.sum(tta_output, dim=0) / len(tta_output)  # = average/mean aggregation

        else:
            raise ValueError('step must be \'train\', \'val\' or \'test\'')

        return output
