from __future__ import annotations
import cv2
import numpy as np

def letterbox(
    img: np.ndarray,
    new_shape: int = 640,
    color=(114, 114, 114),
):
    """
    Resize+pad to square new_shape while keeping aspect ratio.
    Returns: img_out, ratio, (dw, dh)
    """
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))

    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    dw = new_shape - nw
    dh = new_shape - nh
    dw /= 2
    dh /= 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img_out = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img_out, r, (left, top)

def bgr_to_tensor(img_bgr: np.ndarray) -> np.ndarray:
    """
    BGR uint8 -> float32 CHW normalized 0..1 RGB
    """
    img_rgb = img_bgr[:, :, ::-1]
    x = img_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # CHW
    return x