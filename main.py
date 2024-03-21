from utils.face_detection import FaceDetection
from utils.image_util import variance_of_laplacian
import cv2, os, time


if __name__ == '__main__':
    start_time = time.time()

    filename = 'Okazaki_Momoko_Girls_Planet_999_profile_photo_29.webp'
    filepath = os.path.join('sample', filename)

    img = cv2.imread(filepath)
    
    face_detection = FaceDetection()
    detect_face = face_detection.auto_rotate_image(img)
    if detect_face['is_face_available'] is False:
        print('Face is not available')
        img = detect_face['image']
        cv2.putText(img, 'Face not detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        print('Face detected')
        img = detect_face['image']
        bounding_box = detect_face['bounding_box']
        cv2.rectangle(img, (bounding_box[0], bounding_box[1]), (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (0, 255, 0), 2)

    output_filename = os.path.basename(filepath).split('.')[0] + '-output.jpg'
    output_filepath = os.path.join('sample', output_filename)
    cv2.imwrite(output_filepath, img)

    blur_score = variance_of_laplacian(img)
    print('Blur score: {:.2f}'.format(blur_score))
    print('Elapsed time {:.2f} s'.format(time.time() - start_time))