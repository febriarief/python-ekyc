import cv2, numpy as np


def variance_of_laplacian(img: np.ndarray) -> float:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(img, cv2.CV_64F).var()