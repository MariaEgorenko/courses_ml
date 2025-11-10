import face_recognition
import cv2
import numpy as np
import os # Добавим os для работы с файлами

# --- Шаг 1: Обучение (Загрузка известных лиц) ---

# Директория с изображениями для обучения
KNOWN_FACES_DIR = 'lesson20/src' # Используем текущую директорию
TOLERANCE = 0.6 # Допуск (чем ниже, тем строже сравнение)

known_face_encodings = []
known_face_names = []

print("Загрузка известных лиц...")

# Ищем файлы .jpg и .png в директории
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        
        # Убираем расширение файла, чтобы использовать как имя
        name = os.path.splitext(filename)[0] 
        
        # Пропускаем сам скрипт, если он в той же папке :)
        if name == "webcam_rec": 
            continue

        try:
            # Загружаем изображение
            image = face_recognition.load_image_file(os.path.join(KNOWN_FACES_DIR, filename))
            # Получаем кодировку (предполагаем, что на фото одно лицо)
            encoding = face_recognition.face_encodings(image)[0]
            
            # Добавляем кодировку и имя в наши списки
            known_face_encodings.append(encoding)
            known_face_names.append(name)
            print(f"Загружено лицо: {name}")

        except IndexError:
            print(f"Внимание: На изображении {filename} не найдено лиц.")
        except Exception as e:
            print(f"Ошибка при обработке {filename}: {e}")


print("Обучение завершено.")

# --- Шаг 2: Инициализация видеопотока ---




# Получаем доступ к веб-камере (0 - обычно встроенная камера)
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Ошибка: не удалось получить доступ к камере.")
    exit()

# --- Шаг 3: Цикл обработки видео (Распознавание) ---

while True:
    # 1. Захватываем один кадр из видео
    ret, frame = video_capture.read()

    if not ret:
        print("Ошибка: не удалось прочитать кадр.")
        break

    # 2. Оптимизация: Уменьшаем кадр для ускорения обработки (например, в 4 раза)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # 3. Конвертируем BGR (OpenCV) в RGB (face_recognition)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # 4. Находим все лица на *уменьшенном* кадре
    face_locations = face_recognition.face_locations(rgb_small_frame)
    # 5. Получаем кодировки для найденных лиц
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # 6. Сравниваем каждое найденное лицо с нашими известными лицами
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=TOLERANCE)
        name = "Unknown" # По умолчанию "Неизвестен"

        # 7. Ищем лучшее совпадение
        # Используем face_distance, чтобы найти наиболее "близкое" лицо
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    # --- Шаг 4: Отрисовка результатов ---

    # Мы обрабатывали уменьшенный кадр, но рисовать будем на оригинальном (frame)
    # Поэтому нужно масштабировать координаты обратно
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Умножаем на 4 (т.к. уменьшали в 4 раза)
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Устанавливаем цвет рамки
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255) # Зеленый для известных, красный для неизвестных

        # Рисуем рамку вокруг лица
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Рисуем плашку для имени
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        # Пишем имя
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Показываем результат в окне
    cv2.imshow('Video', frame)

    # Выход из цикла по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Шаг 5: Очистка ---
video_capture.release()
cv2.destroyAllWindows()