# -*- coding: utf-8 -*-
"""Boundary IoU (mBIoU) for semantic segmentation.

This module follows the official Boundary IoU definition of
Cheng et al., "Boundary IoU: Improving Object-Centric Image Segmentation
Evaluation", CVPR 2021.  For a binary mask M and a boundary width d, the
boundary region is obtained by subtracting the eroded mask from the mask:

    B(M) = M \\ E_d(M),

which is the definition used by the official boundary-iou-api
(mask_to_boundary: mask - erode(mask)).  The Boundary IoU of a class c in an
image is

    BIoU_c = |B(P_c) & B(G_c)| / |B(P_c) | B(G_c)|,

and mBIoU first averages the image-level scores of each class and then
averages over the classes with valid evaluation results.  Label value 255 is
treated as "ignore" and excluded from the computation.

The erosion uses a 3x3 minimum filter from Pillow, applied d times with a
zero-padded border, i.e. the same successive 3x3 erosions as the official
OpenCV implementation.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageOps

__all__ = ["mask_to_boundary", "boundary_iou_per_image"]


def mask_to_boundary(mask, d):
    """Return the boundary region ``mask & ~erode(mask)``.

    Parameters
    ----------
    mask : array-like
        Binary mask of shape (H, W).
    d : int
        Erosion width in pixels, ``max(1, round(r * image_diagonal))``.
    """
    mask = np.asarray(mask, dtype=bool)
    d = max(1, int(round(float(d))))
    h, w = mask.shape
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    padded = ImageOps.expand(img, border=d, fill=0)
    for _ in range(d):
        padded = padded.filter(ImageFilter.MinFilter(3))
    eroded = np.asarray(padded, dtype=np.uint8)[d:d + h, d:d + w] > 0
    return mask & ~eroded


def boundary_iou_per_image(pred, target, num_classes, ignore_index=255,
                           dilation_ratio=0.02):
    """Compute the Boundary IoU of every class present in one image.

    Parameters
    ----------
    pred : array-like
        Predicted label map, shape (H, W) or (1, H, W).
    target : array-like
        Ground-truth label map, shape (H, W) or (1, H, W).
    num_classes : int
        Number of classes.
    ignore_index : int
        Label value excluded from the evaluation (default: 255).
    dilation_ratio : float
        Boundary width ratio r (default: 0.02, as in the paper).

    Returns
    -------
    dict
        ``{class_id: boundary_iou}`` for classes present in the ground truth
        (classes with zero ground-truth pixels are skipped).
    """
    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.ndim == 3:
        pred = pred[0]
    if target.ndim == 3:
        target = target[0]
    if pred.shape != target.shape:
        raise ValueError(
            "pred and target must have the same shape, got {} and {}".format(
                pred.shape, target.shape))

    h, w = target.shape
    d = max(1, int(round(float(dilation_ratio) * float(np.hypot(h, w)))))

    valid = target != ignore_index
    scores = {}
    for c in np.unique(target[valid]):
        c = int(c)
        if c < 0 or c >= num_classes:
            continue
        gt = (target == c) & valid
        if not gt.any():
            continue
        pred_c = (pred == c) & valid
        b_pred = mask_to_boundary(pred_c, d)
        b_gt = mask_to_boundary(gt, d)
        union = np.count_nonzero(b_pred | b_gt)
        if union == 0:
            # Both boundary regions are empty: Boundary IoU is defined as 1.
            scores[c] = 1.0
        else:
            scores[c] = float(np.count_nonzero(b_pred & b_gt)) / float(union)
    return scores
