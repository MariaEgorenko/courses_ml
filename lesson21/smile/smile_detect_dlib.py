import dlib
import cv2
import numpy as np

def euclidean_distance(p1, p2):
    """Рассчитывает евклидово расстояние между двумя точками dlib"""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def is_smiling(landmarks, threshold=0.6):
    """
    Определяет улыбку, нормализуя ширину рта (48, 54)
    относительно расстояния между внешними уголками глаз (36, 45).
    """
    p36 = landmarks.part(36)  # внешний угол левого глаза 
    p45 = landmarks.part(45)  # внешний угол правого глаза
    eye_distance = euclidean_distance(p36, p45)

    p48 = landmarks.part(48)  # левый внешний угол губ
    p54 = landmarks.part(54)  # правый внешний угол губ
    mouth_width = euclidean_distance(p48, p54)

    ratio = mouth_width / eye_distance
    # print(ratio)  # отладка
    return ratio > threshold

predictor = dlib.shape_predictor('lesson21/models/shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('\033[31mНе удалось получить доступ к камере...\033[0m')
    exit()

while True:
    ret, img = cap.read()

    if not ret:
        print('\033[31mНе удалось прочитать кадр...\033[0m')
        break

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray_img)

    for face in faces:
        landmarks = predictor(gray_img, face)


        for n in range(48, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(img, (x, y), 1, (0, 225, 0), -1)

        if is_smiling(landmarks):
            cv2.putText(img, 'Smile', (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 225, 0), 2)

    cv2.imshow('Find smile', img)
    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()