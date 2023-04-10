from pathlib import Path

import numpy as np
from PIL import Image
from skimage.filters import gaussian, laplace


def load_image() -> np.ndarray:
    """Load the example image into memory."""
    filepath = Path(__file__).parents[1] / "assets" / "image1.jpg"
    return np.array(Image.open(filepath))


def denoise(image: np.ndarray) -> np.ndarray:
    """Denoise the image by applying a gaussian blur."""
    return gaussian(image, sigma=1, channel_axis=-1)


def detect_edges(image: np.ndarray) -> np.ndarray:
    """Detect edges in the image by applying a laplace filter."""
    return laplace(image, ksize=3)


def calculate_sharpness_score(image: np.ndarray) -> float:
    """Calculate the sharpness score of the image."""
    return np.median(np.abs(image)).astype(float)


def laplace_of_gaussian_sharpness(image: np.ndarray) -> float:
    """Calculate the laplace of gaussian of the image.

    This implementation denoises the image, applies an edge detection filter and
    takes the median value of resulting edges as a sharpness metric.
    """
    image_denoised = denoise(image)
    image_log = laplace(image_denoised)
    sharpness_score = calculate_sharpness_score(image_log)
    return sharpness_score


if __name__ == "__main__":
    image = load_image()
    sharpness_score = laplace_of_gaussian_sharpness(image)
    print(f"Sharpness score: {sharpness_score:.2f}")
