"""Bounding-box utilities for weakly-supervised object localization (CUB).

Given a foreground score map (e.g. ``1 - p(background)`` from the seg head, or a
binary pseudo-mask), derive a predicted bounding box and compare it against the
ground-truth box with IoU.  Used by both the CUB training-time validation and
the standalone test script.
"""

import cv2
import numpy as np


def normalize_scoremap(scoremap):
    """Min-max normalise a 2-D score map to [0, 1] (NaN-safe)."""
    scoremap = scoremap.astype(np.float32)
    if np.isnan(scoremap).any():
        scoremap = np.nan_to_num(scoremap, nan=0.0)
    vmin, vmax = scoremap.min(), scoremap.max()
    if vmax <= vmin:
        return np.zeros_like(scoremap)
    return (scoremap - vmin) / (vmax - vmin)


def scoremap_to_bbox(scoremap, threshold=0.5, multi_contour=False):
    """Extract a bbox ``[x0, y0, x1, y1]`` from a (already [0,1]) score map.

    Threshold is relative: pixels above ``threshold`` are treated as foreground.
    Returns the tightest box around the largest connected component (or the
    union of all components when ``multi_contour`` is True).  Falls back to the
    full-image box when nothing passes the threshold.
    """
    h, w = scoremap.shape
    binary = (scoremap >= threshold).astype(np.uint8)
    if binary.sum() == 0:
        return np.array([0, 0, w - 1, h - 1], dtype=np.float32)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.array([0, 0, w - 1, h - 1], dtype=np.float32)

    if multi_contour:
        boxes = [cv2.boundingRect(c) for c in contours]
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes)
        y1 = max(b[1] + b[3] for b in boxes)
    else:
        c = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(c)
        x0, y0, x1, y1 = x, y, x + bw, y + bh

    return np.array([x0, y0, x1, y1], dtype=np.float32)


def bbox_iou(box_a, box_b):
    """IoU of two ``[x0, y0, x1, y1]`` boxes."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b

    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih

    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)
