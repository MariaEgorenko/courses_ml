import cv2
import mediapipe as mp
import math

# --- Вспомогательные функции ---

def get_distance(p1, p2):
    """Рассчитывает евклидово расстояние между двумя точками."""
    return math.sqrt(((p1[0] - p2[0])**2) + ((p1[1] - p2[1])**2))

def get_ear(lm_list, eye_indices, image_w, image_h):
    """
    Рассчитывает коэффициент пропорциональности глаза (EAR) по 6 точкам.
    
    lm_list: Список всех 468 точек лица
    eye_indices: Индексы 6 точек для одного глаза
    """
    # 
    
    # Преобразуем нормализованные координаты в пиксели
    p1 = (lm_list[eye_indices[0]].x * image_w, lm_list[eye_indices[0]].y * image_h)
    p2 = (lm_list[eye_indices[1]].x * image_w, lm_list[eye_indices[1]].y * image_h)
    p3 = (lm_list[eye_indices[2]].x * image_w, lm_list[eye_indices[2]].y * image_h)
    p4 = (lm_list[eye_indices[3]].x * image_w, lm_list[eye_indices[3]].y * image_h)
    p5 = (lm_list[eye_indices[4]].x * image_w, lm_list[eye_indices[4]].y * image_h)
    p6 = (lm_list[eye_indices[5]].x * image_w, lm_list[eye_indices[5]].y * image_h)

    # Рассчитываем вертикальные расстояния
    vert_dist1 = get_distance(p2, p6)
    vert_dist2 = get_distance(p3, p5)

    # Рассчитываем горизонтальное расстояние
    horiz_dist = get_distance(p1, p4)
    
    if horiz_dist == 0:
        return 0.0

    # Рассчитываем EAR
    ear = (vert_dist1 + vert_dist2) / (2.0 * horiz_dist)
    return ear

# --- Инициализация ---

# Индексы MediaPipe для 6 точек вокруг каждого глаза (P1-P6)
# P1, P4 - горизонтальные края
# P2, P3 - верхнее веко
# P5, P6 - нижнее веко

# ПРАВЫЙ глаз (с точки зрения человека)
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
# ЛЕВЫЙ глаз (с точки зрения человека)
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Порог EAR. Если EAR ниже этого значения, считаем глаз закрытым.
# Возможно, вам придется его немного подстроить!
EYE_EAR_THRESHOLD = 0.2

cap = cv2.VideoCapture(0)

# --- Основной цикл ---

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Получаем размеры кадра
    image_h, image_w, _ = image.shape
    
    # Конвертируем BGR в RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Обработка MediaPipe
    image_rgb.flags.writeable = False
    results = face_mesh.process(image_rgb)
    image_rgb.flags.writeable = True
    
    # Обратно в BGR для OpenCV
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    status = "Eyes Open"
    
    if results.multi_face_landmarks:
        # Получаем точки для первого (и единственного) лица
        face_landmarks = results.multi_face_landmarks[0]
        lm = face_landmarks.landmark
        
        # Рассчитываем EAR для правого глаза
        right_ear = get_ear(lm, RIGHT_EYE_INDICES, image_w, image_h)
        
        # Рассчитываем EAR для левого глаза
        left_ear = get_ear(lm, LEFT_EYE_INDICES, image_w, image_h)
        
        # Если ОБА глаза закрыты (ниже порога)
        if right_ear < EYE_EAR_THRESHOLD and left_ear < EYE_EAR_THRESHOLD:
            status = "Eyes Closed"

    # --- Отображение ---
    
    # Переворачиваем кадр для "зеркального" отображения (как в зеркале)
    flipped_image = cv2.flip(image_bgr, 1)

    # Устанавливаем цвет текста в зависимости от статуса
    color = (0, 0, 255) if status == "Eyes Closed" else (0, 255, 0)
    
    # Рисуем статус на перевернутом кадре
    cv2.putText(flipped_image, status, 
                (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, color, 2, cv2.LINE_AA)

    # Показываем результат
    cv2.imshow('MediaPipe Eye Detector', flipped_image)

    # Выход по нажатию 'ESC'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
face_mesh.close()