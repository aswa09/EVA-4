from albumentations import (
    Compose,
    PadIfNeeded,
    RandomCrop,
    HorizontalFlip,
    Cutout,
    HueSaturationValue,
    Rotate,
    Normalize
)

from albumentations.pytorch import ToTensor
import numpy as np


def albumentations_train_transforms(mean, std, p=1.0):
    transforms_list = []

    transforms_list.extend([
        PadIfNeeded(min_height=40, min_width=40, border_mode=4, value=None, p=1.0),
        RandomCrop(32, 32),
        HorizontalFlip(p=0.2),
        Cutout(8,8,p=0.1), #num_holes=1, max_h_size=8, max_w_size=8, always_apply=False, p=1.0
        HueSaturationValue(p=0.25),
        Rotate(limit=15),
        Normalize(
            mean=mean,
            std=std,
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensor()

    ])

    transforms = Compose(transforms_list, p=p)
    return lambda img: transforms(image=np.array(img))["image"]