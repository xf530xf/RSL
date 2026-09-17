"""CUB-200-2011 dataset loaders for the RSL WSOL pipeline.

Loading style follows CCAM/WSOL (image list + per-image class label + GT
bounding box from the official annotation files), wrapped in the same loader
interface used by the VOC / underwater RSL loaders so the existing model and
training loop run unchanged.

Layout assumptions
------------------
root_dir/                                  (the CUB_200_2011 image root)
    images/<class>/<file>.jpg

name_list_dir/                             (datasets/cub, produced by
    train.txt        image stems (relative path w/o .jpg)   prepare_cub_splits.py)
    val.txt          test-split image stems
    cls_labels_onehot.npy   {stem: uint8[num_classes]}  one-hot over 1..200
    boxes.npy        {stem: float32[x0, y0, x1, y1]}  GT box (orig pixels)
    category.txt     200 species names

Label convention (matches underwater/VOC loaders):
    channel 0       = background
    channels 1..200 = bird species
"""

import os

import imageio.v2 as imageio
import numpy as np
from torch.utils.data import Dataset

from . import transforms


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


def load_box_list(name_list_dir):
    return np.load(
        os.path.join(name_list_dir, 'boxes.npy'),
        allow_pickle=True,
    ).item()


def robust_read_image(image_name):
    image = np.asarray(imageio.imread(image_name))
    if image.ndim < 3:
        image = np.stack((image, image, image), axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image


class CUBDataset(Dataset):
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
        self.name_list_path = os.path.join(name_list_dir, split + '.txt')
        self.name_list = load_img_name_list(self.name_list_path)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        stem = str(self.name_list[idx])
        img_path = os.path.join(self.img_dir, stem + '.jpg')
        image = robust_read_image(img_path)
        return stem, image


class CUBClsDataset(CUBDataset):
    """Training loader: returns the (weak) image-level class label + crop box."""

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
        num_classes=201,
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
                image = transforms.random_scaling(image, scale_range=self.rescale_range)
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
        stem, image = super().__getitem__(idx)
        image, img_box = self.__transforms(image=image)

        cls_label = self.label_list[stem]

        if self.aug:
            return stem, image, cls_label, img_box
        return stem, image, cls_label


class CUBEvalDataset(CUBDataset):
    """Evaluation loader for weakly-supervised object localization.

    Returns the (normalised, original-resolution) image, the GT bounding box in
    original-pixel coords ``[x0, y0, x1, y1]`` and the (GT-known) class label.
    """

    def __init__(
        self,
        root_dir=None,
        name_list_dir=None,
        split='val',
        stage='val',
        ignore_index=255,
        num_classes=201,
        **kwargs,
    ):
        super().__init__(root_dir, name_list_dir, split, stage, ignore_index)
        self.num_classes = num_classes
        self.label_list = load_cls_label_list(name_list_dir=name_list_dir)
        self.box_list = load_box_list(name_list_dir=name_list_dir)

    def __getitem__(self, idx):
        stem, image = super().__getitem__(idx)
        h, w = image.shape[0], image.shape[1]

        image = transforms.normalize_img(image)
        image = np.transpose(image, (2, 0, 1))

        box = np.asarray(self.box_list[stem], dtype=np.float32)  # x0, y0, x1, y1
        cls_label = self.label_list[stem]
        size = np.asarray([w, h], dtype=np.int32)

        return stem, image, box, cls_label, size
