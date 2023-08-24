import random

import torch
from torch import nn

# uncomment to debug
# import matplotlib.pyplot as plt


class SegmentationWrapper(nn.Module):
    def __init__(self, model, augmentations):
        super().__init__()
        self.model = model
        self.product = augmentations.product
        self.list = augmentations.list
        self.delist = augmentations.delist

    def augment_inputs_batch(self, inputs, params_batch):
        augmented_inputs = {key: [] for key in inputs.keys()}

        for i, params in enumerate(params_batch):

            inputs_i = {key: inputs[key][i].clone() for key in inputs.keys()}

            # uncomment to debug
            # plt.imshow(inputs_i['aerial'][:3].permute(1, 2, 0))
            # plt.show()

            for augmentation, param in zip(self.list, params):
                inputs_i = augmentation.augment(inputs_i, param)

            augmented_inputs = {key: augmented_inputs[key] + [inputs_i[key]] for key in inputs.keys()}

        augmented_inputs = {key: torch.stack(augmented_inputs[key]) for key in inputs.keys()}

        return augmented_inputs

    def deaugment_outputs_batch(self, outputs, deparams_batch):
        deaugmented_outputs = []

        for i, deparams in enumerate(deparams_batch):
            output_i = outputs[i].clone()

            for deaugmentation, de_param in zip(self.delist, deparams):
                output_i = deaugmentation.deaugment(output_i, de_param.clone())

            deaugmented_outputs.append(output_i)

        deaugmented_outputs = torch.stack(deaugmented_outputs)

        return deaugmented_outputs

    def forward(self, inputs, step, batch_size, limit=None, random_seed=42):
        # step can be "training", "validation", "test" or "predict"
        random.seed(random_seed)

        if step == 'training':
            # augment the inputs
            params_batch = random.choices(self.product, k=batch_size)
            inputs = self.augment_inputs_batch(inputs, params_batch)

            # process the inputs in the model
            outputs = self.model(**inputs)

            # deaugment the model outputs
            deparams_batch = [params[::-1] for params in params_batch]
            outputs = self.deaugment_outputs_batch(outputs, deparams_batch)

        elif step == 'validation' or step == 'test' or step == 'predict':
            tta_params = self.product if limit is None else random.choices(self.product, k=limit)
            tta_deparams = [p[::-1] for p in self.product]
            outputs = []

            for i in range(batch_size):
                inputs_i = {key: inputs[key][i] for key in inputs.keys()}
                tta_inputs_i = {key: [] for key in inputs_i.keys()}

                # uncomment to debug
                # plt.imshow(inputs_i['aerial'][:3].permute(1, 2, 0))
                # plt.show()

                for params in tta_params:
                    augmented_inputs_i = inputs_i.copy()

                    for augmentation, param in zip(self.list, params):
                        augmented_inputs_i = augmentation.augment(augmented_inputs_i, param)

                    for key in inputs.keys():
                        tta_inputs_i[key].append(augmented_inputs_i[key])

                for key in tta_inputs_i.keys():
                    tta_inputs_i[key] = torch.stack(tta_inputs_i[key], dim=0)

                tta_outputs_i = self.model(**tta_inputs_i)

                for i, deparams in enumerate(tta_deparams):
                    for deaugmentation, de_param in zip(self.delist, deparams):
                        tta_outputs_i[i] = deaugmentation.deaugment(tta_outputs_i[i], de_param)

                outputs_i = torch.sum(tta_outputs_i, dim=0) / len(tta_outputs_i)

                # uncomment to debug
                # plt.imshow(outputs_i[:3].permute(1, 2, 0))
                # plt.show()

                outputs.append(outputs_i)

            outputs = torch.stack(outputs)

        else:
            raise ValueError('step must be "training", "validation", "test" or "predict"')

        return outputs
