import cv2

model_path = 'lesson20/src/models/cv2_trainer.yml'

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade_db = cv2.CascadeClassifier(cascade_path)

names = ['who is it', 'Mari', 'Putin']

video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Ошибка: не удалось получить доступ к камере.")
    exit()

while True:
    ret, img = video_capture.read()

    if not ret:
        print("Ошибка: не удалось прочитать кадр.")
        break

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade_db.detectMultiScale(img_gray, 1.1, 5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(img_gray[y:y+h, x:x+w])
        if (confidence < 75):
            try:
                id_name = names[id] 
                color = (255, 255, 255)
            except IndexError:
                id_name = "Unknown name"
                color = (0, 255, 255)
        else:
            id_name = "who is it"
            color = (0, 0, 255)

        text = f"{id_name} ({round(confidence, 1)})"
        cv2.putText(img, text, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('Webcam Recognition', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()