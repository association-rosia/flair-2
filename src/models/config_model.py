from typing import Any

class FLAIR2ConfigModel():
    def __init__(
        self,
        arch_lib: str,
        arch: str,
        encoder_name: str,
        classes: list,
        learning_rate: float,
        class_weights: list,
        list_images_train: list,
        list_images_val: list,
        list_images_test: list,
        aerial_list_bands: list,
        sen_size: int,
        sen_temp_size: int,
        sen_temp_reduc: int,
        sen_list_bands: list,
        prob_cover: float,
        use_augmentation: bool,
        use_tta: bool,
        one_vs_all: int,
        train_batch_size: int,
        test_batch_size: int,
        *args: Any,
        **kwargs: Any
    ) -> None:
        
        self.arch_lib = arch_lib
        self.arch = arch
        self.encoder_name = encoder_name
        self.classes = classes
        self.learning_rate = learning_rate
        self.class_weights = class_weights
        self.list_images_train = list_images_train
        self.list_images_val = list_images_val
        self.list_images_test = list_images_test
        self.aerial_list_bands = aerial_list_bands
        self.sen_size = sen_size
        self.sen_temp_size = sen_temp_size
        self.sen_temp_reduc = sen_temp_reduc
        self.sen_list_bands = sen_list_bands
        self.prob_cover = prob_cover
        self.use_augmentation = use_augmentation
        self.use_tta = use_tta
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.one_vs_all = one_vs_all