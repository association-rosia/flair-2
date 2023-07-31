from torch import nn
import random


class SegmentationWrapper(nn.Module):
    def __init__(self, model, augmentations):
        super().__init__(model, augmentations)
        self.model = model
        self.augmentations = augmentations

    def forward(self, inputs, step, limit=None):  # limit the number of augmentations for TTA
        # step can be 'train', 'val, 'test'

        if step == 'train':
            params = random.choice(self.augmentations.product)
            augmentations = self.augmentations.list

            augmented_inputs = []
            for input in inputs:
                for augmentation, param in zip(augmentations, params):
                    augmented_input = augmentation.apply(input, param)
                    augmented_inputs.append(augmented_input)

            output = self.model(*augmented_inputs)

        elif step == 'val' or step == 'test':
            if limit is not None:
                product = random.choices(self.augmentations.product, k=limit)
            else:
                product = self.augmentations.product

            augmentations = self.augmentations.list

            # TODO: do a better approach
            augmented_inputs = []
            for input in inputs:
                for params in product:
                    for augmentation, param in zip(augmentations, params):
                        augmented_input = augmentation.apply(input, param)
                        augmented_inputs.append(augmented_input)


        else:
            raise ValueError('step must be \'train\', \'val\' or \'test\'')

        return output
