import numpy as np
import cv2

cascade_face_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascaade = cv2.CascadeClassifier(cascade_face_path)

cascade_smile_path = cv2.data.haarcascades + 'haarcascade_smile.xml'
smile_cascade = cv2.CascadeClassifier(cascade_smile_path)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('\033[31mНе удалось получить доступ к камере...\033[0m')
    exit()

while True :
    ret, img = cap.read()

    if not ret:
        print('\033[31mНе удалось прочитать кадр...\033[0m')
        break

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascaade.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray_img[y:y+h, x:x+h]
        
        smile = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(25, 25)
        )

        for i in smile:
            if len(smile) > 1:
                cv2.putText(img, 'Smiling', (x, y-30), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (0, 255, 0), 3, cv2.LINE_AA)
                
    cv2.imshow('Find smile', img)
    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()