from __future__ import annotations
import cv2
import numpy as np
import torch

def scale_boxes_back(
    boxes_xyxy: torch.Tensor,
    ratio: float,
    pad: tuple[int, int],
    orig_shape: tuple[int, int],
):
    """
    Map boxes from letterboxed image back to original image coordinates.
    pad = (pad_x, pad_y) = (left, top)
    """
    pad_x, pad_y = pad
    b = boxes_xyxy.clone()
    b[:, [0, 2]] -= pad_x
    b[:, [1, 3]] -= pad_y
    b /= ratio

    h, w = orig_shape
    b[:, 0].clamp_(0, w)
    b[:, 2].clamp_(0, w)
    b[:, 1].clamp_(0, h)
    b[:, 3].clamp_(0, h)
    return b

def draw_detections(
    img_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    names: dict[int, str] | None = None,
):
    out = img_bgr.copy()
    for (x1, y1, x2, y2), s, c in zip(boxes_xyxy, scores, classes):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = names.get(int(c), str(int(c))) if names else str(int(c))
        text = f"{label} {float(s):.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, text, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out