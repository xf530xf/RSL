"""Generate train/val splits, per-image one-hot class labels, and a cleaned
category list for the AquaOV255 underwater dataset.

Outputs (under datasets/underwater/):
  train.txt
  val.txt
  cls_labels_onehot.npy   {img_stem: np.uint8[num_classes]}  (one-hot over fg ids)
  category.txt            254 fg class names (cleaned, one per line)

Mask convention in AquaOV255:
  - 0-indexed class id (cls_id - 1) for foreground pixels
  - 65535 for ignore / unannotated pixels
  - exactly one foreground class per image

Adopted convention here (matches the rest of the new underwater pipeline):
  - num_classes = 255  (index 0 = background, indices 1..254 = the 254 fg classes)
  - cls_labels_onehot is sized 255; the bg slot (index 0) is always 0.
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm


NUM_FG_CLASSES = 254
NUM_CLASSES = NUM_FG_CLASSES + 1  # +1 for background
MASK_IGNORE_RAW = 65535
# Highest raw value that maps to a *named* fg class.  A handful of masks in the
# release contain raw value 254 (4 files: Catfish_112, Lanternfish_001..003)
# which is one past the last entry in category.txt — treat those as ignore.
MAX_VALID_RAW = NUM_FG_CLASSES - 1  # i.e. 253


def _clean_category_file(src_path, dst_path):
    """Read the upstream category.txt and write a cleaned newline-separated list."""
    with open(src_path, 'r') as f:
        names = [line.strip().strip(']').strip() for line in f if line.strip()]
    if len(names) != NUM_FG_CLASSES:
        raise RuntimeError(
            f"Expected {NUM_FG_CLASSES} class names in {src_path}, got {len(names)}"
        )
    with open(dst_path, 'w') as f:
        for n in names:
            f.write(n + '\n')
    return names


def _extract_fg_id(mask_path):
    """Return the unique 1-indexed foreground class id present in this mask.

    Raw values that exceed ``MAX_VALID_RAW`` are treated as ignore (they
    correspond to a handful of dataset annotation glitches with no named
    class).  Returns ``None`` if the mask has no usable fg pixels.
    """
    m = np.array(Image.open(mask_path))
    valid = (m != MASK_IGNORE_RAW) & (m <= MAX_VALID_RAW)
    if not valid.any():
        return None
    vals = np.unique(m[valid])
    if vals.size > 1:
        flat = m[valid]
        unique, counts = np.unique(flat, return_counts=True)
        fg_raw = int(unique[np.argmax(counts)])
        return fg_raw + 1
    return int(vals[0]) + 1


def build(data_root, out_dir, val_ratio=0.2, seed=1):
    images_dir = os.path.join(data_root, 'images')
    masks_dir = os.path.join(data_root, 'masks')
    os.makedirs(out_dir, exist_ok=True)

    _clean_category_file(
        os.path.join(data_root, 'category.txt'),
        os.path.join(out_dir, 'category.txt'),
    )

    img_files = sorted(
        f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')
    )
    if not img_files:
        raise RuntimeError(f"No .jpg files found under {images_dir}")

    onehot = {}
    by_class = defaultdict(list)
    missing = []

    for fname in tqdm(img_files, desc='scanning masks', ncols=100):
        stem = os.path.splitext(fname)[0]
        mask_path = os.path.join(masks_dir, stem + '.png')
        if not os.path.isfile(mask_path):
            missing.append(stem)
            continue
        fg_id = _extract_fg_id(mask_path)
        if fg_id is None:
            missing.append(stem)
            continue
        vec = np.zeros(NUM_CLASSES, dtype=np.uint8)
        vec[fg_id] = 1
        onehot[stem] = vec
        by_class[fg_id].append(stem)

    if missing:
        print(f"[warn] skipped {len(missing)} images without usable masks")

    rng = np.random.RandomState(seed)
    train_list, val_list = [], []
    for cid in sorted(by_class.keys()):
        stems = sorted(by_class[cid])
        rng.shuffle(stems)
        n = len(stems)
        n_val = int(round(n * val_ratio))
        if n >= 2 and n_val == 0:
            n_val = 1
        if n >= 2 and n_val == n:
            n_val = n - 1
        val_list.extend(stems[:n_val])
        train_list.extend(stems[n_val:])

    train_list.sort()
    val_list.sort()

    with open(os.path.join(out_dir, 'train.txt'), 'w') as f:
        for s in train_list:
            f.write(s + '\n')
    with open(os.path.join(out_dir, 'val.txt'), 'w') as f:
        for s in val_list:
            f.write(s + '\n')

    np.save(os.path.join(out_dir, 'cls_labels_onehot.npy'), onehot, allow_pickle=True)

    print(f"total images:    {len(img_files)}")
    print(f"with cls label:  {len(onehot)}")
    print(f"train samples:   {len(train_list)}")
    print(f"val samples:     {len(val_list)}")
    print(f"classes present: {len(by_class)} / {NUM_FG_CLASSES}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='/path/to/AquaOV255')
    p.add_argument('--out_dir', default='datasets/underwater')
    p.add_argument('--val_ratio', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--force', action='store_true',
                   help='regenerate even if outputs already exist')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    required = [
        os.path.join(args.out_dir, 'train.txt'),
        os.path.join(args.out_dir, 'val.txt'),
        os.path.join(args.out_dir, 'cls_labels_onehot.npy'),
        os.path.join(args.out_dir, 'category.txt'),
    ]
    if not args.force and all(os.path.isfile(p) for p in required):
        print('All split artifacts already exist; pass --force to regenerate.')
        sys.exit(0)
    build(args.data_root, args.out_dir, val_ratio=args.val_ratio, seed=args.seed)
