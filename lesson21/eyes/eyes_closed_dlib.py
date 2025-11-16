import dlib
import cv2
import numpy as np

def euclidean_distance(p1, p2):
    """Рассчитывает евклидово расстояние между двумя точками dlib"""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def calculate_ear(eye_landmarks):
    v1 = euclidean_distance(eye_landmarks[1], eye_landmarks[5])
    v2 = euclidean_distance(eye_landmarks[2], eye_landmarks[4]) 

    h = euclidean_distance(eye_landmarks[0], eye_landmarks[3]) 

    if h == 0:
        return 0.0

    ear = (v1 + v2) / (2.0 * h)
    return ear

def is_eyes_closed(landmarks, threshold=0.17):

    left_eye_landmarks = [landmarks.part(i) for i in range(36, 42)]
    right_eye_landmarks = [landmarks.part(i) for i in range(42, 48)]

    ear_left = calculate_ear(left_eye_landmarks)
    ear_right = calculate_ear(right_eye_landmarks)

    ear = (ear_right + ear_left) / 2

    return ear < threshold

predictor = dlib.shape_predictor("lesson21/models/shape_predictor_68_face_landmarks.dat")
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

        for n in range(36, 42):
            cv2.circle(img, (landmarks.part(n).x, landmarks.part(n).y), 1, (0, 255, 0), -1)
        for n in range(42, 48):
            cv2.circle(img, (landmarks.part(n).x, landmarks.part(n).y), 1, (0, 255, 0), -1)

        if is_eyes_closed(landmarks):
            cv2.putText(img, 'Eyes Closed', (face.left(), face.top() - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
    cv2.imshow('video', img)
    key = cv2.waitKey(30) & 0xff
    if key == 27: # ESC
        break

cap.release()
cv2.destroyAllWindows()