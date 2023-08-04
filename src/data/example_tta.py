from torch import nn
from torch.utils.data import DataLoader

from src.data.tta import augmentations, wrappers
from src.data.make_dataset import get_list_images, FLAIR2Dataset
import src.constants as cst


class FakeModel(nn.Module):

    def forward(self, aerial, sen):
        return aerial


model = FakeModel()

augmentations = augmentations.Augmentations([
    augmentations.HorizontalFlip(),
    augmentations.VerticalFlip(),
    augmentations.Rotate([90, 180, 270]),
])

tta_wrapper = wrappers.SegmentationWrapper(model, augmentations)

path_train = cst.PATH_DATA_TRAIN
list_images_train = get_list_images(path_train)

dataset_train = FLAIR2Dataset(
    list_images=list_images_train,
    sen_size=40,
    is_test=False,
)

batch_size = 16
dataloader_train = DataLoader(
    dataset=dataset_train,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)

image_id, aerial, sen, labels = next(iter(dataloader_train))
output = tta_wrapper(inputs={'aerial': aerial, 'sen': sen}, step='validation', batch_size=batch_size)
