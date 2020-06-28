from albumentations import (
    Compose,
    HorizontalFlip,
    VerticalFlip,
    Cutout,
    HueSaturationValue,
    Rotate,
    Normalize
)

from albumentations.pytorch import ToTensor
import numpy as np
import sys


def albumentations_train_transforms(args, p=1.0):
    try:
        transforms_list = []

        # if args.cutout_prob > 0:  # CutOut
        #     if isinstance(args.mean, float):
        #         fvalue = args.mean * 255.0
        #     else:
        #         fvalue = tuple([x * 255.0 for x in args.mean])
        #(num_holes=1, max_h_size=args.cutout_dim[0], max_w_size=args.cutout_dim[1], fill_value=fvalue,p=args.cutout_prob)
        #p=args.cutout_prob, num_holes=1, fill_value=fill_value, max_h_size=args.cutout_dim[0],max_width=args.cutout_dim[1]
        transforms_list.extend([
            HorizontalFlip(p=args.hflip_prob),
            VerticalFlip(p=args.vflip_prob),
            Cutout(num_holes=1, max_h_size=args.cutout_dim[0], max_w_size=args.cutout_dim[1],p=args.cutout_prob),
            HueSaturationValue(p=args.hue_val),
            Rotate(limit=args.rotate_lim),
            Normalize(
                mean=args.mean,
                std=args.std,
                p=1.0
            ),
            ToTensor()

        ])

        transforms = Compose(transforms_list, p=p)
        return lambda img: transforms(image=np.array(img))["image"]
    except Exception as e:
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno) + " " + type(e).__name__ + " " + str(e))
        sys.exit(1)
