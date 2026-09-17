"""Class-name and background-prompt lists for the AquaOV255 underwater dataset.

The 254 foreground class names are loaded from
`datasets/underwater/category.txt` (produced by
`scripts/prepare_underwater_splits.py`).  Each `CamelCase` name is converted to
a CLIP-friendly lower-case form, e.g. `BlueRingedOctopus` -> `blue ringed
octopus`, `Crocodile&Alligator` -> `crocodile alligator`.

A small underwater-scene background prompt list is provided for the same
`zeroshot_classifier` pipeline that `BACKGROUND_CATEGORY` is used for in VOC.
"""

import os
import re


_DEFAULT_CATEGORY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'datasets', 'underwater', 'category.txt',
)


def _camel_to_words(name):
    name = name.replace('&', ' ').replace('-', ' ').replace('_', ' ')
    spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    spaced = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', spaced)
    spaced = re.sub(r'\s+', ' ', spaced).strip()
    return spaced.lower()


def _load_class_names(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Underwater category file not found: {path}. "
            f"Run scripts/prepare_underwater_splits.py first."
        )
    raw_names = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip().strip(']').strip()
            if line:
                raw_names.append(line)
    if len(raw_names) != 254:
        raise RuntimeError(
            f"Expected 254 class names in {path}, got {len(raw_names)}"
        )
    return raw_names, [_camel_to_words(n) for n in raw_names]


raw_class_names_underwater, class_names_underwater = _load_class_names(
    _DEFAULT_CATEGORY_PATH
)


BACKGROUND_CATEGORY_UNDERWATER = [
    'water', 'sea', 'ocean', 'sea water', 'underwater scene',
    'sand', 'seabed', 'sea floor', 'sea bottom', 'mud',
    'rock', 'rocks', 'pebbles', 'gravel',
    'coral reef', 'reef',
    'sea grass', 'sea weed', 'algae', 'kelp',
    'sun light', 'sunbeam', 'caustics', 'light ray',
    'bubble', 'bubbles', 'foam',
    'plankton', 'particle', 'silt', 'haze',
    'sky', 'cloud',
    'boat hull', 'pier',
    'diver bubbles',
]
