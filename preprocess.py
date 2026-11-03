"""Image preprocessing shared by the web server and the tests.

The browser sends a clean, white-on-black picture of the drawn digit. This
module turns it into the same kind of 28x28 image MNIST uses: the digit is
scaled to fit a 20x20 box, placed on a 28x28 canvas, and shifted so its centre
of mass sits in the middle. Matching MNIST here is what keeps live accuracy
close to the accuracy you see during training.

This file deliberately does not import torch.
"""
import cv2
import numpy as np

# MNIST pixel statistics, used to normalise inputs in training and inference.
MEAN, STD = 0.1307, 0.3081


def to_mnist(gray: np.ndarray, threshold: int = 32):
    """Return a 28x28 uint8 image, or None if the input has no ink."""
    ys, xs = np.nonzero(gray > threshold)
    if xs.size == 0:
        return None

    crop = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    img = np.zeros((28, 28), np.uint8)
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    img[top:top + new_h, left:left + new_w] = crop

    m = cv2.moments(img)
    if m["m00"] > 0:
        # Whole-pixel shifts only: sub-pixel shifts would blur the strokes.
        dx = int(np.rint(13.5 - m["m10"] / m["m00"]))
        dy = int(np.rint(13.5 - m["m01"] / m["m00"]))
        shift = np.float32([[1, 0, dx], [0, 1, dy]])
        img = cv2.warpAffine(img, shift, (28, 28), flags=cv2.INTER_NEAREST)
    return img


def normalise(img28: np.ndarray) -> np.ndarray:
    """uint8 28x28 -> float32 array shaped (1, 1, 28, 28), normalised for the CNN."""
    x = img28.astype(np.float32) / 255.0
    return ((x - MEAN) / STD)[None, None]
