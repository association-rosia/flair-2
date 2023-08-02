import torch
from torch import nn
import random


class SegmentationWrapper(nn.Module):
    def __init__(self, model, augmentations):
        super().__init__()
        self.model = model
        self.product = augmentations.product
        self.list = augmentations.list
        self.delist = augmentations.delist
        # self.aggregation = aggregation

    def forward(self, inputs, step, batch_size, limit=None, random_seed=42):  # TODO : manage batch size
        # step can be "training", "validation", "test" or "predict"
        random.seed(random_seed)

        if step == 'training':
            params = random.choice(self.product)

            for augmentation, param in zip(self.list, params):
                inputs = augmentation.augment(inputs, param)

            output = self.model(**inputs)

            deparams = params[::-1]
            for deaugmentation, de_param in zip(self.delist, deparams):
                output = deaugmentation.deaugment(output, de_param)

        elif step == 'validation' or step == 'test' or step == 'predict':
            product = self.product if limit is None else random.choices(self.product, k=limit)
            deproduct = [p[::-1] for p in self.product]

            tta_inputs = {key: [] for key in inputs.keys()}
            for params in product:

                augmented_inputs = inputs.copy()

                for augmentation, param in zip(self.list, params):
                    augmented_inputs = augmentation.augment(augmented_inputs, param)

                for key in inputs.keys():
                    tta_inputs[key].append(augmented_inputs[key])

            for key in tta_inputs.keys():
                tta_inputs[key] = torch.stack(tta_inputs[key], dim=0)

            tta_output = self.model(**tta_inputs)

            for i, deparams in enumerate(deproduct):

                for deaugmentation, de_param in zip(self.delist, deparams):
                    tta_output[i] = deaugmentation.deaugment(tta_output[i], de_param)

            output = torch.sum(tta_output, dim=0) / len(tta_output)  # = average/mean aggregation
            # TODO: add more aggregation possibilities

        else:
            raise ValueError('step must be "training", "validation", "test" or "predict"')

        return output
