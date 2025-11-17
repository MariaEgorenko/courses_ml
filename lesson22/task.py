import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import random
import torch

point_pairs = [
    (5, 11), (6, 12), # Торс
    (11, 12), # Между ног
    (0, 1), (0, 2), (1, 3), (2, 4),  # Голова
    (5, 6),  # Соединение плечей
    (5, 7), (7, 9),  # Левая рука
    (6, 8), (8, 10),  # Правая рука
    (11, 13), (13, 15),  # Нога левая
    (12, 14), (14, 16)  # Нога правая
]


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

cascade_face_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascaade = cv2.CascadeClassifier(cascade_face_path)

def video_tracking(model, video_path, output_video_path="output_video.mp4"):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print('\033[31mНе удалось открыть видео...\033[0m')
        exit()

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    track_history = {}
    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print('\033[31mНе удалось прочитать кадр...\033[0m')
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = model.track(frame, iou=0.5, conf=0.3, persist=True, imgsz=608, verbose=False, tracker="botsort.yaml")

        if results[0].boxes.id != None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, id in zip(boxes, ids):
                
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0

                speed_px_sec = 0.0
    
                if id in track_history:
                    last = track_history[id]
                    last_cx, last_cy = last["last_center"]
                    last_frame = last["last_frame"]

                    dt_frames = frame_idx - last_frame
                    if dt_frames > 0:
                        dx = cx - last_cx
                        dy = cy - last_cy
                        dist_px = np.sqrt(dx * dx + dy * dy)  # пиксели за dt_frames
                        # скорость в пикселях в секунду
                        speed_px_sec_instant = (dist_px / dt_frames) * fps

                        # Немного сгладим (эксп. среднее)
                        alpha = 0.5
                        speed_px_sec = alpha * speed_px_sec_instant + (1 - alpha) * last["speed"]
                    else:
                        speed_px_sec = last["speed"]
                else:
                    speed_px_sec = 0.0

                # обновляем историю
                track_history[id] = {
                    "last_center": (cx, cy),
                    "last_frame": frame_idx,
                    "speed": speed_px_sec,
                }
                random.seed(int(id))
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3],), color, 2)
                cv2.putText(
                    frame,
                    f"Id {id}",
                    (box[0], box[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )

                people_img = gray[box[1]:box[3], box[0]:box[2]]
                faces = face_cascaade.detectMultiScale(
                    people_img,
                    scaleFactor=1.1,
                    minNeighbors=1,
                    minSize=(25, 25)
                )

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, 
                                 (box[0] + x, box[1] + y),  # Верхний левый угол с отступом
                                 (box[0] + x + w, box[1] + y + h), # Нижний правый угол с отступом
                                 (0, 255, 0), 
                                 2)


            for i, id in enumerate(ids):
                keypoints = results[0].keypoints.xy[i]

                random.seed(int(id))
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                # Рисуем точки
                for j, point in enumerate(keypoints):
                    x, y = point
                    if x > 0 and y > 0:  # Проверяем, что обе координаты больше нуля
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)

                # Соединяем точки линиями
                for pair in point_pairs:
                    start, end = pair
                    if keypoints[start][0] > 0 and keypoints[start][1] > 0 and keypoints[end][0] > 0 and keypoints[end][1] > 0:  # Проверяем, что обе пары координат больше нуля
                        x1, y1 = int(keypoints[start][0]), int(keypoints[start][1])
                        x2, y2 = int(keypoints[end][0]), int(keypoints[end][1])
                        cv2.line(frame, (x1, y1), (x2, y2), color, 2)


        out.write(frame)

    cap.release()
    out.release()

    cv2.destroyAllWindows()

video_path = 'lesson22/src/in/video_1.mp4'
output_path = 'lesson22/src/out/video_1.mp4'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = YOLO('yolov8m-pose.pt')  # load a pretrained model (recommended for training)
model = model.to(device)
video_tracking(model, video_path, output_video_path=output_path)