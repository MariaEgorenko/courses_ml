import face_recognition
import cv2
import numpy as np
import os

KNOWN_FACES_DIR = 'lesson20/src/face'
TOLERANCE = 0.6  # допуск (чем ниже, тем строже сравнение)

known_face_encodings = []
known_face_names = []

print("Загрузка известных лиц...\n")

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        
        name = os.path.splitext(filename)[0] 
        
        try:
            # загрузка изображения
            image = face_recognition.load_image_file(os.path.join(KNOWN_FACES_DIR, filename))
            # получение коддировки
            # на фото должно быть одно лицо
            encoding = face_recognition.face_encodings(image)[0]
            
            # добавление кодировки и имени в списки
            known_face_encodings.append(encoding)
            known_face_names.append(name)
            print(f"Загружено лицо: {name}")

        except IndexError:
            print(f"Не найдено лиц на фото {filename}")
        except Exception as e:
            print(f"Ошибка при обработке {filename}: {e}")


print("Загрузка лиц завершена\n")

# инициализация видеопотока 

# доступ к веб-камере
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Ошибка: не удалось получить доступ к камере.")
    exit()

# обработка видео

while True:
    # захват кадра
    ret, frame = video_capture.read()

    if not ret:
        print("Ошибка: не удалось прочитать кадр.")
        break

    # уменьшениие кадра для ускорения обработки (в 4 раза)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # сравнение лиц с известными
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=TOLERANCE)
        name = "who is it" # по умолчанию

        # поиск совпадений
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    # отрисовка результатов

    # масштабирование координаты
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Умножаем на 4, т.к. уменьшали в 4 раза
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # цвет рамки
        color = (0, 255, 0) if name != "who is it" else (255, 128, 0) # Зеленый для известных, красный для неизвестных

        # отрисовка рамки
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # плашка для имени
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    # выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# очистка
video_capture.release()
cv2.destroyAllWindows()