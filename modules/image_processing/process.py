from django.conf import settings
from PIL import Image, ImageFile
from helpers import utils
import numpy as np
import base64, cv2, dlib, io, traceback, uuid

def compress_image(base64_image):
    compressed_image_path =  "{}/storages/images".format(settings.BASE_DIR)

    try:
        random_string = str(uuid.uuid4())
        compressed_filename = random_string + "-COMPRESSED.png"
        compressed_filepath = "{}/{}".format(compressed_image_path, compressed_filename)
    
        img = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_image, 'utf-8'))))
        img = auto_rotate_image(img)

        width = int(img.width * (300 / img.height))
        img.thumbnail((width, 300), Image.ANTIALIAS)
        img.save(compressed_filepath)

        return { "status": 200, "filename": compressed_filename, "filepath": compressed_filepath }
    
    except Exception as e:
        utils.remove_image(compressed_image_path)
        utils.create_log(traceback.format_exc())
        return { "status": 500, "data": {}, "message": str(e) }

def auto_rotate_image(image: ImageFile) -> ImageFile:
    rotated_image = image
    
    for cycle in range(0, 4):
        if cycle > 0:
            rotated_image = rotated_image.rotate(90, expand=True)

        image_copy = np.asarray(rotated_image)
        image_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

        detector = dlib.get_frontal_face_detector()
        faces = detector(image_gray, 0)

        if cycle < 3 and len(faces) == 0:
            continue

        if cycle == 3 and len(faces) == 0:
            return image

        return rotated_image