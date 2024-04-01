from utils.config import get_config
from utils.face_detection import FaceDetection
from utils.image_util import resize, variance_of_laplacian
from resources.anti_spoof.anti_spoof import AntiSpoof, CropImage, parse_model_name
import argparse, cv2, numpy as np, os, sys, time


MODEL_DIR = './resources/models/spoof_detection_models'


if __name__ == '__main__':
    desc = 'Face liveness detection'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--filename', type=str, default=None, help='image used to test')
    args = parser.parse_args()

    if args.filename is None:
        print('Please provide args --filename')
        sys.exit()

    start_time = time.time()

    filename = args.filename
    filepath = os.path.join('sample', filename)
    img = cv2.imread(filepath)

    height, width = img.shape[0], img.shape[1]
    if width / height != 3/4:
        print('Image aspect ratio is not 3:4')
        sys.exit()

    conf = get_config()
    max_height = conf.face_detection_size
    if height > max_height:
        img = resize(img)
    
    face_detection = FaceDetection()
    detect_face = face_detection.detect_face_align(img)

    if detect_face['is_face_available'] is False:
        print('Face is not available')
        img = detect_face['image']
        cv2.putText(img, 'Face not detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        print('Face detected')
        
        img = detect_face['image']
        bounding_box = detect_face['bounding_box']
        
        anti_spoof = AntiSpoof(0)
        image_cropper = CropImage()
        prediction = np.zeros((1, 3))
        
        for model_name in os.listdir(MODEL_DIR):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                'org_img': img,
                'bbox': bounding_box,
                'scale': scale,
                'out_w': w_input,
                'out_h': h_input,
                'crop': True
            }

            if scale is None:
                param["crop"] = False

            img_cropped = image_cropper.crop(**param)
            prediction += anti_spoof.predict(img_cropped, os.path.join(MODEL_DIR, model_name))

        label = np.argmax(prediction)
        value = prediction[0][label]/2
        if label == 1:
            print("Image '{}' is Real Face. \nScore: {:.2f}.".format(filename, value))
            result_text = "Real Face Score: {:.2f}".format(value)
            color = (255, 0, 0)
        else:
            print("Image '{}' is Fake Face. \nScore: {:.2f}.".format(filename, value))
            result_text = "Fake Face Score: {:.2f}".format(value)
            color = (0, 0, 255)

        cv2.rectangle(img, (bounding_box[0], bounding_box[1]), (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), color, 2)
        cv2.putText(img, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    output_filename = os.path.basename(filepath).split('.')[0] + '-output.jpg'
    output_filepath = os.path.join('sample', output_filename)
    cv2.imwrite(output_filepath, img)

    blur_score = variance_of_laplacian(img)
    print('Blur score: {:.2f}'.format(blur_score))

    end_time = time.time()
    print('Elapsed time {:.2f} s'.format(end_time - start_time))