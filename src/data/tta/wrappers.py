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

            # plt.imshow(inputs['aerial'][:3, :, :].permute(1, 2, 0))
            # plt.show()

            for augmentation, param in zip(self.list, params):
                inputs = augmentation.apply(inputs, param)

            output = self.model(**inputs)

            # plt.imshow(output[:3, :, :].permute(1, 2, 0))
            # plt.show()

            de_params = params[::-1]
            for de_augmentation, de_param in zip(self.de_list, de_params):
                output = de_augmentation.de_apply(output, de_param)

            # plt.imshow(output[:3, :, :].permute(1, 2, 0))
            # plt.show()

        elif step == 'val' or step == 'test':
            product = self.product if limit is None else random.choices(self.product, k=limit)
            de_product = product[::-1]

            plt.imshow(inputs['aerial'][:3, :, :].permute(1, 2, 0))
            plt.show()

            tta_inputs = {key: [] for key in inputs.keys()}
            for params in product:
                for augmentation, param in zip(self.list, params):
                    inputs = augmentation.apply(inputs, param)

                for key in inputs.keys():
                    tta_inputs[key].append(inputs[key])

            for key in tta_inputs.keys():
                tta_inputs[key] = torch.stack(tta_inputs[key], dim=0)

            tta_output = self.model(**tta_inputs)
            for i, de_params in enumerate(de_product):
                for de_augmentation, de_param in zip(self.de_list, de_params):
                    tta_output[i] = de_augmentation.de_apply(tta_output[i], de_param)

            output = torch.sum(tta_output, dim=0) / len(tta_output)  # = average/mean aggregation

            plt.imshow(output[:3, :, :].permute(1, 2, 0))
            plt.show()

        else:
            raise ValueError('step must be \'train\', \'val\' or \'test\'')

        return output
