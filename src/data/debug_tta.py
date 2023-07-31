from torch import nn

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
    # augmentations.Solarize([0, 0.25, 0.5, 0.75])
])

wrapper = wrappers.SegmentationWrapper(model, augmentations)

path_train = cst.PATH_DATA_TRAIN
list_images_train = get_list_images(path_train)

dataset_train = FLAIR2Dataset(
    list_images=list_images_train,
    sen_size=40,
    is_test=False,
    use_augmentation=False,
)

image_id, aerial, sen, labels = dataset_train[0]

output = wrapper(inputs={'aerial': aerial, 'sen': sen}, step='test')  # use as model in the loop
