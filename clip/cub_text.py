"""Class-name and background-prompt lists for the CUB-200-2011 dataset.

The 200 foreground (bird species) names are loaded from
``datasets/cub/category.txt`` (produced by ``scripts/prepare_cub_splits.py``).
Each ``001.Black_footed_Albatross``-style name is normalised to a CLIP-friendly
lower-case form, e.g. ``Black_footed_Albatross`` -> ``black footed albatross``.

A small natural-scene background prompt list is provided for the same
``zeroshot_classifier`` pipeline that ``BACKGROUND_CATEGORY`` is used for in VOC.
"""

import os
import re

_DEFAULT_CATEGORY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'datasets', 'cub', 'category.txt',
)


def _name_to_words(name):
    name = name.replace('&', ' ').replace('-', ' ').replace('_', ' ').replace('.', ' ')
    # strip a possible leading numeric id ("001 Black footed Albatross")
    name = re.sub(r'^\s*\d+\s*', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name.lower()


def _load_class_names(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"CUB category file not found: {path}. "
            f"Run scripts/prepare_cub_splits.py first."
        )
    raw_names = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                raw_names.append(line)
    if len(raw_names) != 200:
        raise RuntimeError(
            f"Expected 200 class names in {path}, got {len(raw_names)}"
        )
    return raw_names, [_name_to_words(n) for n in raw_names]


raw_class_names_cub, class_names_cub = _load_class_names(_DEFAULT_CATEGORY_PATH)


# Generic natural-scene background prompts (birds appear in the wild / captive
# scenes); used to build the background text features for CAM derivation.
BACKGROUND_CATEGORY_CUB = [
    'sky', 'cloud', 'clouds',
    'tree', 'trees', 'branch', 'branches', 'twig', 'trunk',
    'leaf', 'leaves', 'foliage', 'bush', 'shrub',
    'grass', 'lawn', 'meadow', 'field', 'reed', 'reeds',
    'flower', 'flowers',
    'ground', 'soil', 'dirt', 'mud', 'sand',
    'rock', 'rocks', 'stone', 'gravel',
    'water', 'lake', 'river', 'sea', 'ocean', 'pond', 'wave', 'waves',
    'snow', 'ice',
    'wood', 'fence', 'post', 'pole', 'wire', 'cable',
    'building', 'wall', 'roof',
    'feeder', 'cage', 'net',
]
