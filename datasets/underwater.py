"""AquaOV255 underwater dataset loaders.

Layout assumptions
------------------
root_dir/
    images/           *.jpg
    masks/            *.png   (single channel, values in {cls_id - 1, 65535})

name_list_dir/
    train.txt         one image stem per line
    val.txt           one image stem per line
    cls_labels_onehot.npy   {stem: np.uint8[num_classes]}  (one-hot over fg ids)
    category.txt      254 fg names

Mask remap performed here
-------------------------
    raw_value 65535 -> ignore_index (default 255)
    raw_value cls_id - 1 -> cls_id   (so 0 stays free for "background")

Final label range after remap: {0 unused | 1..254 fg | 255 ignore}.
"""

import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from . import transforms


MASK_IGNORE_RAW = 65535
# Highest raw mask value that maps to a *named* fg class id (1..254).  Any raw
# value above this is also treated as ignore — a few release-time annotation
# glitches contain raw value 254 which has no entry in category.txt.
MAX_VALID_RAW = 253


def load_img_name_list(img_name_list_path):
    img_name_list = np.loadtxt(img_name_list_path, dtype=str)
    if img_name_list.ndim == 0:
        img_name_list = np.array([str(img_name_list)])
    return img_name_list


def load_cls_label_list(name_list_dir):
    return np.load(
        os.path.join(name_list_dir, 'cls_labels_onehot.npy'),
        allow_pickle=True,
    ).item()


def robust_read_image(image_name):
    image = np.asarray(imageio.imread(image_name))
    if image.ndim < 3:
        image = np.stack((image, image, image), axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image


def _remap_mask(raw_mask, ignore_index=255):
    """Convert raw AquaOV255 mask to the RSL label convention.

    Foreground pixels become 1..254, anything else (raw 65535, or a stray raw
    value > 253) becomes ``ignore_index``.  Returns int16 to allow values up
    to 255.
    """
    label = raw_mask.astype(np.int32)
    out = np.full(label.shape, ignore_index, dtype=np.int16)
    fg = (label != MASK_IGNORE_RAW) & (label <= MAX_VALID_RAW)
    out[fg] = (label[fg] + 1).astype(np.int16)
    return out


class UnderwaterDataset(Dataset):
    def __init__(
        self,
        root_dir=None,
        name_list_dir=None,
        split='train',
        stage='train',
        ignore_index=255,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.stage = stage
        self.ignore_index = ignore_index
        self.img_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'masks')
        self.name_list_path = os.path.join(name_list_dir, split + '.txt')
        self.name_list = load_img_name_list(self.name_list_path)

    def __len__(self):
        return len(self.name_list)

    def _read_label(self, stem):
        label_path = os.path.join(self.label_dir, stem + '.png')
        raw = np.array(Image.open(label_path))
        return _remap_mask(raw, ignore_index=self.ignore_index)

    def __getitem__(self, idx):
        stem = str(self.name_list[idx])
        img_path = os.path.join(self.img_dir, stem + '.jpg')
        image = robust_read_image(img_path)

        if self.stage in ('train', 'val'):
            label = self._read_label(stem)
        else:
            label = image[:, :, 0]
        return stem, image, label


class UnderwaterClsDataset(UnderwaterDataset):
    def __init__(
        self,
        root_dir=None,
        name_list_dir=None,
        split='train',
        stage='train',
        resize_range=(512, 640),
        rescale_range=(0.5, 2.0),
        crop_size=320,
        img_fliplr=True,
        ignore_index=255,
        num_classes=255,
        aug=False,
        **kwargs,
    ):
        super().__init__(root_dir, name_list_dir, split, stage, ignore_index)
        self.aug = aug
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.img_fliplr = img_fliplr
        self.num_classes = num_classes
        self.color_jittor = transforms.PhotoMetricDistortion()

        self.label_list = load_cls_label_list(name_list_dir=name_list_dir)

    def __transforms(self, image):
        img_box = None
        if self.aug:
            image = np.array(image)
            if self.rescale_range:
                image = transforms.random_scaling(
                    image,
                    scale_range=self.rescale_range,
                )
            if self.img_fliplr:
                image = transforms.random_fliplr(image)
            if self.crop_size:
                image, img_box = transforms.random_crop(
                    image,
                    crop_size=self.crop_size,
                    mean_rgb=[0, 0, 0],
                    ignore_index=self.ignore_index,
                )
        image = transforms.normalize_img(image)
        image = np.transpose(image, (2, 0, 1))
        return image, img_box

    def __getitem__(self, idx):
        stem, image, _ = super().__getitem__(idx)
        image, img_box = self.__transforms(image=image)

        cls_label = self.label_list[stem]

        if self.aug:
            return stem, image, cls_label, img_box
        return stem, image, cls_label


class UnderwaterSegDataset(UnderwaterDataset):
    def __init__(
        self,
        root_dir=None,
        name_list_dir=None,
        split='train',
        stage='train',
        resize_range=(512, 640),
        rescale_range=(0.5, 2.0),
        crop_size=320,
        img_fliplr=True,
        ignore_index=255,
        num_classes=255,
        aug=False,
        **kwargs,
    ):
        super().__init__(root_dir, name_list_dir, split, stage, ignore_index)
        self.aug = aug
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.img_fliplr = img_fliplr
        self.num_classes = num_classes
        self.color_jittor = transforms.PhotoMetricDistortion()

        self.label_list = load_cls_label_list(name_list_dir=name_list_dir)

    def __transforms(self, image, label):
        if self.aug:
            image = np.array(image)
            if self.img_fliplr:
                image, label = transforms.random_fliplr(image, label)
            image = self.color_jittor(image)
            if self.crop_size:
                image, label, img_box = transforms.random_crop(
                    image,
                    label,
                    crop_size=self.crop_size,
                    ignore_index=self.ignore_index,
                )
        image = transforms.normalize_img(image)
        image = np.transpose(image, (2, 0, 1))
        return image, label

    def __getitem__(self, idx):
        stem, image, label = super().__getitem__(idx)
        image, label = self.__transforms(image=image, label=label)

        if self.stage == 'test':
            cls_label = np.zeros(self.num_classes, dtype=np.uint8)
        else:
            cls_label = self.label_list[stem]

        return stem, image, label, cls_label
