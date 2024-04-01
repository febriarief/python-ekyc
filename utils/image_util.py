from utils.config import get_config
import cv2, numpy as np, math


def variance_of_laplacian(img: np.ndarray) -> float:
    """Calculate variance of Laplacian of the image.

    Args:
        img: Original image as a NumPy array.

    Returns:
        Variance of Laplacian of the image.
    """

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(img, cv2.CV_64F).var()


def resize(img: np.ndarray) -> np.ndarray:
    """Resize image to the specified size.

    Args:
        img: Original image as a NumPy array.

    Returns:
        Resized image as a NumPy array.
    """

    conf = get_config()
    target = conf.face_detection_size
    aspect_ratio = 3/4
    new_height = target
    new_width = int(new_height * aspect_ratio)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return img


def resize_with_pad(img: np.ndarray) -> np.ndarray:
    """Resize image with padding to maintain aspect ratio.

    Args:
        img: Original image as a NumPy array.

    Returns:
        Resized image with padding as a NumPy array.
    """

    conf = get_config()
    face_detection_size = conf.face_detection_size
    min_face_detection_size = conf.min_face_detection_size
    
    target_size = (face_detection_size, face_detection_size)
    min_target_size = min_face_detection_size
    height, width = img.shape[:2]
    aspect_ratio = width / height

    if width * height >= min_target_size * min_target_size:
        img = cv2.resize(img, (int(min_target_size * math.sqrt(aspect_ratio)), int(min_target_size / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)
        height, width = img.shape[:2]
        
    new_height, new_width = target_size
    pad_left = pad_right = pad_top = pad_bottom = 0

    if width < new_width:
        pad_left = int((new_width - width) / 2)
        pad_right = int(new_width - width - pad_left)

    if height < new_height:
        pad_top = int((new_height - height) / 2)
        pad_bottom = int(new_height - height - pad_top)

    padded_img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_img