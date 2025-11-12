import cv2
import numpy as np
import os

FACE_DIR = 'lesson20/src/face/'

cascade_path = cv2.data.haarcascades+'haarcascade_frontalface_default.xml'

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade_db = cv2.CascadeClassifier(cascade_path)

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]
    face_samples = []
    ids = []

    for image_path in image_paths:
        img = cv2.imread(image_path)

        if img is None:
            print(f"не удалось прочитать изображение {image_path}")
            continue
        try:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            filename = os.path.split(image_path)[-1]
            id = int(filename.split('.')[1])

            faces = face_cascade_db.detectMultiScale(img_gray, 1.1, 5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_samples.append(img_gray[y:y+h, x:x+w])
                ids.append(id)

            if faces.shape[0] == 0:
                print(f"На изображении {filename} не обнаружено лиц")
            
        except Exception as e:
            print(f"ошибка при обработке {image_path}: {e}")
            continue

    return face_samples, ids

faces, ids = get_images_and_labels(FACE_DIR)

recognizer.train(faces, np.array(ids))

out_path = 'lesson20/src/models/'
model = 'cv2_trainer.yml'
recognizer.write(os.path.join(out_path, model))

