from django.conf import settings
from PIL import Image
from helpers import utils
import base64, io, traceback, uuid

def compress_image(base64_image):
    compressed_image_path =  "{}/storages/images".format(settings.BASE_DIR)

    try:
        random_string = str(uuid.uuid4())
        compressed_filename = random_string + "-COMPRESSED.jpg"
        compressed_filepath = "{}/{}".format(compressed_image_path, compressed_filename)
    
        img = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_image, 'utf-8'))))
        width = int(img.width * (200 / img.height))
        img.thumbnail((width, 200), Image.ANTIALIAS)
        img.save(compressed_filepath)

        return { "status": 200, "filename": compressed_filename, "filepath": compressed_filepath }
    
    except Exception as e:
        utils.remove_image(compressed_image_path)
        utils.create_log(traceback.format_exc())
        return { "status": 500, "data": {}, "message": str(e) }
