from .config import get_config
from typing import Any, Dict
import cv2, math, numpy as np, os


class FaceDetection:
    def __init__(self):
        conf = get_config()
        
        curr_path = os.path.abspath(os.path.dirname(__file__))
        caffemodel = os.path.join(curr_path, '..', 'resources', 'face_detection', 'Widerface-RetinaFace.caffemodel')
        deploy = os.path.join(curr_path, '..', 'resources', 'face_detection', 'deploy.prototxt')

        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        self.confidence_threshold = conf.face_conf_thres
        self.face_detection_size = conf.face_detection_size
        self.min_face_detection_size = conf.min_face_detection_size

    # TODO: Rotate the image to detect face in the correct orientation
    def auto_rotate_image(self, img: np.ndarray) -> Dict[str, Any]:
        """Auto rotate the image to detect face in the correct orientation.
        
        Args:
            img: Original image as a NumPy array.
            
        Returns:
            Dictionary containing the following keys:
                - is_face_available: True if face is detected, False otherwise.
                - bounding_box: Bounding box coordinates of the detected face.
                - image: Image with detected face.
        """
        
        org_image = img.copy()

        for _ in range(4):
            detect_face = self.detect_face(img)
            if detect_face['is_face_available'] is True:
                return detect_face
            else:
                img = np.rot90(img)

        return { 'is_face_available': False, 'bounding_box': None, 'image': org_image }

    def detect_face(self, img: np.ndarray) -> Dict[str, Any]:
        """Detect face in the image.

        Args:
            img: Original image as a NumPy array.

        Returns:
            Dictionary containing the following keys:
                - is_face_available: True if face is detected, False otherwise.
                - bounding_box: Bounding box coordinates of the detected face.
                - image: Image with detected face.
        """

        # img = self.resize_with_pad(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        max_conf_index = np.argmax(out[:, 2])
        height, width = img.shape[0], img.shape[1]

        left, top, right, bottom = out[max_conf_index, 3] * width, out[max_conf_index, 4] * height, out[max_conf_index, 5] * width, out[max_conf_index, 6] * height
        bounding_box = [int(left), int(top), int(right - left + 1), int(bottom - top + 1)] # [x, y, w, h]
        
        if out[max_conf_index, 2] > self.confidence_threshold:
            return { 'is_face_available': True, 'bounding_box': bounding_box, 'image': img } 
        else:
            return { 'is_face_available': False, 'bounding_box': None, 'image': None } 
        
    def resize_with_pad(self, img) -> np.ndarray:
        """Resize image with padding to maintain aspect ratio.

        Args:
            img: Original image as a NumPy array.

        Returns:
            Resized image with padding as a NumPy array.
        """
        target_size = (self.face_detection_size, self.face_detection_size)
        min_target_size = self.min_face_detection_size
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
