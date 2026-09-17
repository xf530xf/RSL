"""Prepare CUB-200-2011 metadata for the RSL WSOL pipeline.

Reads the raw CUB-200-2011 annotation files and writes, into
``datasets/cub/``, the small index files the dataset loaders / model expect:

    train.txt               image stems (relative path w/o .jpg), one per line
    val.txt                 test-split image stems, one per line
    category.txt            200 species names (``Black_footed_Albatross`` style)
    cls_labels_onehot.npy   {stem: uint8[num_classes]}  one-hot over 1..200
    boxes.npy               {stem: int32[x0, y0, x1, y1]}  GT box (orig pixels)
    sizes.npy               {stem: int32[width, height]}   original image size

Label convention (matches the underwater/VOC loaders):
    channel 0           = background (never set in cls one-hot)
    channels 1..200     = the 200 bird species

Run once before training::

    python scripts/prepare_cub_splits.py \
        --cub_root /path/to/CUB_200_2011
"""

import argparse
import os

import numpy as np
from PIL import Image

NUM_CLASSES = 201  # background + 200 species


def _read_table(path):
    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(line.split())
    return rows


def main(cub_root, out_dir, read_sizes):
    os.makedirs(out_dir, exist_ok=True)

    # id -> relative image path (e.g. "001.Black_footed_Albatross/xxx.jpg")
    id_to_path = {r[0]: r[1] for r in _read_table(os.path.join(cub_root, 'images.txt'))}
    # id -> class id (1..200)
    id_to_class = {r[0]: int(r[1]) for r in _read_table(os.path.join(cub_root, 'image_class_labels.txt'))}
    # id -> is_train (1 train / 0 test)
    id_to_split = {r[0]: int(r[1]) for r in _read_table(os.path.join(cub_root, 'train_test_split.txt'))}
    # id -> x y w h  (floats, original pixels)
    id_to_box = {r[0]: list(map(float, r[1:5])) for r in _read_table(os.path.join(cub_root, 'bounding_boxes.txt'))}
    # class id -> "001.Black_footed_Albatross"
    classes = _read_table(os.path.join(cub_root, 'classes.txt'))

    # category.txt -> species name with the leading "001." stripped
    category_names = []
    for cid, raw_name in classes:
        name = raw_name.split('.', 1)[-1]
        category_names.append(name)
    with open(os.path.join(out_dir, 'category.txt'), 'w') as f:
        f.write('\n'.join(category_names) + '\n')
    print(f'wrote {len(category_names)} category names')

    train_stems, val_stems = [], []
    cls_onehot, boxes, sizes = {}, {}, {}

    img_root = os.path.join(cub_root, 'images')
    for img_id, rel_path in id_to_path.items():
        stem = os.path.splitext(rel_path)[0]
        class_id = id_to_class[img_id]

        onehot = np.zeros(NUM_CLASSES, dtype=np.uint8)
        onehot[class_id] = 1
        cls_onehot[stem] = onehot

        x, y, w, h = id_to_box[img_id]
        boxes[stem] = np.array([x, y, x + w, y + h], dtype=np.float32)

        if read_sizes:
            with Image.open(os.path.join(img_root, rel_path)) as im:
                sizes[stem] = np.array([im.width, im.height], dtype=np.int32)

        if id_to_split[img_id] == 1:
            train_stems.append(stem)
        else:
            val_stems.append(stem)

    with open(os.path.join(out_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_stems) + '\n')
    with open(os.path.join(out_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_stems) + '\n')

    np.save(os.path.join(out_dir, 'cls_labels_onehot.npy'), cls_onehot)
    np.save(os.path.join(out_dir, 'boxes.npy'), boxes)
    if read_sizes:
        np.save(os.path.join(out_dir, 'sizes.npy'), sizes)

    print(f'train: {len(train_stems)} | val: {len(val_stems)} | classes: {len(category_names)}')
    print(f'outputs written to {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cub_root',
        default='/path/to/CUB_200_2011',
        type=str,
    )
    parser.add_argument(
        '--out_dir',
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'cub'),
        type=str,
    )
    parser.add_argument('--no_sizes', action='store_true',
                        help='skip reading per-image sizes (faster; sizes read at eval instead)')
    args = parser.parse_args()
    main(args.cub_root, args.out_dir, read_sizes=not args.no_sizes)
