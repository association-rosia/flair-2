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
        self.delist = augmentations.delist

    def augment_inputs_batch(self, inputs, params_batch):
        augmented_inputs = {key: [] for key in inputs.keys()}

        for i, params in enumerate(params_batch):

            inputs_i = {key: inputs[key][i] for key in inputs.keys()}

            plt.imshow(inputs_i['aerial'][:3].permute(1, 2, 0))
            plt.show()

            for augmentation, param in zip(self.list, params):
                inputs_i = augmentation.augment(inputs_i, param)

            # plt.imshow(inputs_i['aerial'][:3].permute(1, 2, 0))
            # plt.show()

            augmented_inputs = {key: augmented_inputs[key] + [inputs_i[key]] for key in inputs.keys()}

        augmented_inputs = {key: torch.stack(augmented_inputs[key]) for key in inputs.keys()}

        return augmented_inputs

    def deaugment_output_batch(self, output, deparams_batch):
        for i, deparams in enumerate(deparams_batch):

            # plt.imshow(output[i][:3].permute(1, 2, 0))
            # plt.show()

            for deaugmentation, de_param in zip(self.delist, deparams):
                output[i] = deaugmentation.deaugment(output[i], de_param)

            plt.imshow(output[i][:3].permute(1, 2, 0))
            plt.show()

        return output

    def forward(self, inputs, step, batch_size, limit=None, random_seed=42):  # TODO : manage batch size
        # step can be "training", "validation", "test" or "predict"
        random.seed(random_seed)

        if step == 'training':
            # augment the inputs
            params_batch = [random.choice(self.product) for _ in range(batch_size)]
            inputs = self.augment_inputs_batch(inputs, params_batch)

            # process the inputs in the model
            output = self.model(**inputs)

            # deaugment the model output
            deparams_batch = [params[::-1] for params in params_batch]
            output = self.deaugment_output_batch(output, deparams_batch)

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
