import os
import cv2
import numpy as np
import warnings

from django.conf import settings

from modules.face_detection.src.anti_spoof_predict import AntiSpoofPredict
from modules.face_detection.src.generate_patches import CropImage
from modules.face_detection.src.utility import parse_model_name

warnings.filterwarnings('ignore')

def liveness(filepath):
    model_dir = "{}/modules/face_detection/resources/anti_spoof_models".format(settings.BASE_DIR)
    device_id = 0
    
    try:
        model_test = AntiSpoofPredict(device_id)
        image_cropper = CropImage()
        image = cv2.imread(filepath)
        
        image_bbox = model_test.get_bbox(image)
        prediction = np.zeros((1, 3))
        for model_name in os.listdir(model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": image,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = image_cropper.crop(**param)
            prediction += model_test.predict(img, os.path.join(model_dir, model_name))

        label = np.argmax(prediction)
        livenessResult = True if label == 1 else False
        value = prediction[0][label]/2
        livenessScore = round(value, 2)

        return { "status": 200, "message": "", "data": { "filepath": filepath, "liveness": livenessResult, "score": livenessScore } }
    
    except Exception as e:
        return { "status": 500, "message": str(e), "data": {} } 
