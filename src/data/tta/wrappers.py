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
                output_i = deaugmentation.deaugment(output_i, de_param)

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

        elif step in ['validation', 'test', 'predict']:
            # TODO: force the use of the original image in the TTA
            tta_params = self.product if limit is None else random.choices(self.product, k=limit)
            tta_deparams = [p[::-1] for p in tta_params]
            tta_inputs = []

            # uncomment to debug
            # plt.imshow(inputs['aerial'][0, :3].permute(1, 2, 0))
            # plt.show()

            for params in tta_params:
                inputs_copy = inputs.copy()

                for augmentation, param in zip(self.list, params):
                    inputs_copy = augmentation.augment(inputs_copy, param)

                tta_inputs.append(inputs_copy)

            # uncomment to debug
            # for i in range(limit):
            #     plt.imshow(tta_inputs[i]['aerial'][0, :3].permute(1, 2, 0))
            #     plt.show()

            tta_inputs = {key: [tta_input[key] for tta_input in tta_inputs] for key in tta_inputs[0]}
            tta_inputs = {key: torch.stack(tta_inputs[key]) for key in tta_inputs}

            shape = None
            for key in tta_inputs:
                shape = tta_inputs[key].shape
                tta_inputs[key] = tta_inputs[key].view(shape[0]*shape[1], *shape[2:])

            tta_outputs = self.model(**tta_inputs)
            tta_outputs = tta_outputs.view(shape[0], shape[1], *tta_outputs.shape[1:])

            # tta_outputs = []
            # for i in range(limit):
            #     tta_outputs.append(self.model(**tta_inputs[i]))

            for i, deparams in enumerate(tta_deparams):
                for deaugmentation, de_param in zip(self.delist, deparams):
                    tta_outputs[i] = deaugmentation.deaugment(tta_outputs[i], de_param)

            outputs = torch.mean(tta_outputs, dim=0)

            # uncomment to debug
            # plt.imshow(outputs[0, :3].permute(1, 2, 0))
            # plt.show()

        else:
            raise ValueError('step must be "training", "validation", "test" or "predict"')

        return outputs
