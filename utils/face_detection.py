from .config import get_config
from utils.image_util import resize_with_pad
from typing import Any, Dict
import cv2, dlib, numpy as np, os


class FaceDetection:
    def __init__(self):
        conf = get_config()
        
        curr_path = os.path.abspath(os.path.dirname(__file__))
        caffemodel = os.path.join(curr_path, '..', 'resources', 'models', 'face_detection', 'Widerface-RetinaFace.caffemodel')
        deploy = os.path.join(curr_path, '..', 'resources', 'models', 'face_detection', 'deploy.prototxt')

        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        self.confidence_threshold = conf.face_conf_thres
        self.dlib_shape_predictor = os.path.join(curr_path, '..', 'resources', 'models', 'face_detection', 'shape_predictor_68_face_landmarks.dat')


    def detect_face(self, img: np.ndarray) -> Dict[str, Any]:
        """Detect face in the image.

        Args:
            img: Original image as a NumPy array.

        Returns:
            Dictionary containing the following keys:
                - image: Image with detected face.
                - is_face_available: True if face is detected, False otherwise.
                - bounding_box: Bounding box coordinates of the detected face. Output: [x, y, w, h]
                - face_landmark: Face landmarks (always None). To get face landmark, use detect_face_align() instead.
        """

        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        max_conf_index = np.argmax(out[:, 2])
        height, width = img.shape[0], img.shape[1]

        if out[max_conf_index, 2] > self.confidence_threshold:
            left, top, right, bottom = out[max_conf_index, 3] * width, out[max_conf_index, 4] * height, out[max_conf_index, 5] * width, out[max_conf_index, 6] * height
            bounding_box = [int(left), int(top), int(right - left + 1), int(bottom - top + 1)]
            return { 'image': img, 'is_face_available': True, 'bounding_box': bounding_box, 'face_landmark': None } 
        else:
            return { 'image': img, 'is_face_available': False, 'bounding_box': None, 'face_landmark': None } 
    

    def detect_face_align(self, img: np.ndarray, debug: bool = False) -> Dict[str, Any]:
        """Detect face in the image using dlib and auto align face.
        
        Args:
            img: Original image as a NumPy array.
            debug: If True, draw face landmarks on the image.

        Returns:
            Dictionary containing the following keys:
                - image: Image with detected face.
                - is_face_available: True if face is detected, False otherwise.
                - bounding_box: Bounding box coordinates of the detected face. Output: [x, y, w, h]
                - face_landmark: Face landmarks (always None). To get face landmark, use detect_face_align() instead.
        """

        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        max_conf_index = np.argmax(out[:, 2])
        height, width = img.shape[0], img.shape[1]
        left, top, right, bottom = int(out[max_conf_index, 3] * width), int(out[max_conf_index, 4] * height), \
            int(out[max_conf_index, 5] * width), int(out[max_conf_index, 6] * height)
        
        if out[max_conf_index, 2] < self.confidence_threshold:
            return { 'image': img, 'is_face_available': False, 'bounding_box': None, 'face_landmark': None } 
        
        rectangle = (left, top, right, bottom)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        predictor = dlib.shape_predictor(self.dlib_shape_predictor)
        shape = predictor(gray, dlib.rectangle(*rectangle))
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        eye_left = landmarks[36:42]
        eye_right = landmarks[42:48]
        dY = np.mean(eye_right[:, 1]) - np.mean(eye_left[:, 1])
        dX = np.mean(eye_right[:, 0]) - np.mean(eye_left[:, 0])
        angle = np.degrees(np.arctan2(dY, dX))
        center = (np.mean(landmarks[:, 0]), np.mean(landmarks[:, 1]))
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
        
        if debug is True:
            landmarks = cv2.transform(landmarks.reshape(-1, 1, 2), rotation_matrix).reshape(68, 2)
            for (x, y) in landmarks:
                cv2.circle(img, (int(x), int(y)), 4, (255, 0, 0), -1)

        bounding_box = [left, top, (right - left + 1), (bottom - top + 1)]
        return { 'image': img, 'is_face_available': True, 'bounding_box': bounding_box, 'face_landmark': landmarks } 
