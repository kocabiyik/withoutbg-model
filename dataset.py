import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from configurations import DTYPE, DIM
from mapping import fg_bg_consistency_map
from midas_transforms import Resize, NormalizeImage, PrepareForNet

img_transforms = A.Compose(
    [
        # essential
        A.Resize(height=DIM, width=DIM),
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(DIM, DIM, scale=(0.80, 1.0), p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
        A.RandomGamma(p=0.2),
        # A.Rotate(p=0.5, limit=8),
        # A.ElasticTransform(p=0.5, alpha_affine=10),

        # color distortions
        A.OneOf([
            A.ISONoise(),
            A.HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, always_apply=False),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25),
            A.FancyPCA(),
            A.CLAHE(),
        ], p=1),
    ],
    additional_targets={
        "alpha": "mask",
        "fg": "image",
        "bg": "image",
        "img": "image",
    }
)

# transforms for MiDaS model (inverse depth estimation)  
# source: https://pytorch.org/hub/intelisl_midas_v2/
midas_small_transform = Compose(
    [
        lambda img: {"image": img / 255.0},
        Resize(
            256,
            256,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="upper_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
        # lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
        lambda sample: torch.from_numpy(sample["image"]),
    ]
)


class MattingDataset(Dataset):
    """
    Composite dataset by pasting a foreground to a suitable background.  
    Args:
        data_dir: Directory containing train and val directories
        bg_dir: Directory containing backgrounds. This directory contains subdirectories like outdoor, indoor, highway etc.
        fg_bg_consistency_map: A dictionary contains foreground-background mapping. Example: car - highway
        with_augmentation: Whether augmentation will be applied or not  
        img_transforms: Image transforms  
        midas_transforms: Transforms for MiDaS model. Source: https://pytorch.org/hub/intelisl_midas_v2/

    Returns:
        X: four channels: RGB + depth
        y: binary segment
        X_for_depth: Input for MiDaS model
    """

    def __init__(self,
                 data_dir,
                 bg_dir,
                 consistency_map,
                 with_augmentation=True,
                 img_transform=img_transforms,
                 midas_transforms=midas_small_transform
                 ):

        self.data_dir = data_dir
        self.fg_bg_consistency_map = consistency_map
        self.with_augmentation = with_augmentation
        self.dir_fg = Path(data_dir) / "fg"
        self.dir_bg = bg_dir
        self.fg_file_paths = list(Path(self.dir_fg).rglob('*.png'))
        random.shuffle(self.fg_file_paths)
        self.img_transform = img_transform
        self.midas_transforms = midas_transforms

    def __len__(self):
        return len(self.fg_file_paths)

    def __getitem__(self, index: int):

        # paths
        fp = str(self.fg_file_paths[index])
        path_fg = fp
        thing = fp.split('/')[7]  # object name
        thing = self.fg_bg_consistency_map[thing]
        suitable_bg_dir = random.choices(thing[0], weights=thing[1])[0]
        bg_list = list((Path(self.dir_bg) / suitable_bg_dir).rglob("*.jpg"))
        suitable_bg = random.choice(bg_list)
        path_bg = Path(self.dir_bg) / suitable_bg_dir / suitable_bg

        # foreground
        fg = Image.open(str(path_fg)).convert("RGBA")

        # background
        bg = Image.open(str(path_bg)).convert("RGB")

        # crop maximum area with taking foreground image size
        image_width, image_height = bg.size
        fg_size = fg.size
        ar = fg_size[0] / fg_size[1]
        center_x = int(image_width / 2)
        center_y = int(image_height / 2)

        if ar < 1:
            crop_width = min(image_height * ar, image_width)
            crop_height = crop_width / ar
        else:
            crop_height = min(image_width / ar, image_height);
            crop_width = crop_height * ar

        x = crop_width
        y = crop_height
        left = center_x - int(x / 2)
        top = center_y - int(y / 2)
        right = center_x + int(x / 2)
        bottom = center_y + int(y / 2)

        bg = bg.crop((left, top, right, bottom))
        bg = bg.resize(size=fg.size)

        # image composited
        img_composited = Image.new(size=fg.size, mode="RGBA")
        img_composited.paste(bg)
        img_composited.paste(fg, (0, 0), fg)
        img_composited = img_composited.convert("RGB")

        # resize
        # note: consider Albumentations
        img_composited = img_composited.resize((DIM, DIM))
        fg = fg.resize((DIM, DIM))
        bg = bg.resize((DIM, DIM))

        alpha = fg.split()[-1]

        fg = fg.convert("RGB")

        alpha = np.array(alpha)
        fg = np.array(fg)
        bg = np.array(bg)
        img = np.array(img_composited)

        input_midas = self.midas_transforms(img)

        # augmentations
        if self.with_augmentation:
            transformed = self.img_transform(image=img, fg=fg, bg=bg, alpha=alpha)
            img = transformed["image"]
            alpha = transformed["alpha"]
            fg = transformed["fg"]
            bg = transformed["bg"]

        # image
        img = np.rollaxis(img, 2, 0)
        img = torch.tensor(img)

        fg = np.rollaxis(fg, 2, 0)
        fg = torch.tensor(fg)

        bg = np.rollaxis(bg, 2, 0)
        bg = torch.tensor(bg)

        alpha = torch.tensor(alpha)
        alpha = torch.unsqueeze(alpha, 0)

        # final processing: Normalization and setting the correct data type
        # Note: consider Albumentations
        x = img
        x = x / 255
        x = x.type(DTYPE)
        y = torch.cat(tensors=[alpha, bg, fg, img])
        y = y / 255
        y = y.type(DTYPE)

        return x, y, input_midas
