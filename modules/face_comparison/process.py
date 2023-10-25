import face_recognition

def faceDistance(imgPath1, imgPath2):
    try:
        image1 = face_recognition.load_image_file(imgPath1)
        image1_encoding = face_recognition.face_encodings(image1)[0]

        image2 = face_recognition.load_image_file(imgPath2)
        image2_encoding = face_recognition.face_encodings(image2)[0]

        face_distances = face_recognition.face_distance([image1_encoding], image2_encoding)

        return face_distances
    
    except:
        return []
