from time import time

from torch import nn
from torch.utils.data import DataLoader

from src.constants import get_constants
from src.data.make_dataset import get_list_images, FLAIR2Dataset
from src.data.tta import augmentations as agms
from src.data.tta import wrappers as wrps

cst = get_constants()


class FakeModel(nn.Module):

    def forward(self, aerial, sen):
        return aerial


model = FakeModel()

augmentations = agms.Augmentations([
    agms.HorizontalFlip(),
    agms.VerticalFlip(),
    agms.Rotate([90, 180, 270]),
    agms.Perspective([0.25, 0.5, 0.75])
])

tta_wrapper = wrps.SegmentationWrapper(model, augmentations)

path_train = cst.path_data_train
list_images_train = get_list_images(path_train)

dataset_train = FLAIR2Dataset(
    list_images=list_images_train,
    sen_size=40,
    is_test=False,
)

batch_size = 24
dataloader_train = DataLoader(
    dataset=dataset_train,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)

start = time()

for image_id, aerial, sen, labels in dataloader_train:
    output = tta_wrapper(inputs={'aerial': aerial, 'sen': sen}, step='validation', batch_size=batch_size, limit=10)

end = time()

print(f'Finish with: {end - start} seconds')
