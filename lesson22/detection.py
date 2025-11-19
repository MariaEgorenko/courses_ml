import cv2
from ultralytics import YOLO
import torch
import random
import dlib
import numpy as np

point_pairs_pose = [
    (5, 11), (6, 12), # Торс
    (11, 12), # Между ног
    (0, 1), (0, 2), (1, 3), (2, 4),  # Голова
    (5, 6),  # Соединение плечей
    (5, 7), (7, 9),  # Левая рука
    (6, 8), (8, 10),  # Правая рука
    (11, 13), (13, 15),  # Нога левая
    (12, 14), (14, 16)  # Нога правая
]

predictor = dlib.shape_predictor('lesson22/models/shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()

def draw_keypoints(frame, keypoints, color=(0, 0, 255)):
    # Рисуем линии
    for pair in point_pairs_pose:
        partA_idx, partB_idx = pair

        if partA_idx >= len(keypoints) or partB_idx >= len(keypoints):
            continue

        pt1 = keypoints[partA_idx]
        pt2 = keypoints[partB_idx]

        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]), int(pt2[1])

        if x1 > 5 and y1 > 5 and x2 > 5 and y2 > 5:
            cv2.line(frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

    # 2. Рисуем точки
    for point in keypoints:
        x, y = int(point[0]), int(point[1])
        
        if x > 5 and y > 5:
            cv2.circle(frame, (x, y), 4, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1, lineType=cv2.LINE_AA)
    
    return frame

def face_detected(gray_frame):
    faces = detector(gray_frame)
    if len(faces) < 1:
        return None
    
    faces_coor = []
    faces_landmarks = []

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        landmarks = predictor(gray_frame, face)

        x = max(0, x)
        y = max(0, y)

        faces_coor.append((x, y, w, h))
        faces_landmarks.append(landmarks)

    return faces_coor, faces_landmarks

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
    return ratio > threshold

def video_tracking(model, source=0, show_video=True, save_video=False, output_video_path="output_video.mp4"):

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print('\033[31mНе удалось открыть видео...\033[0m')
        exit()

    if save_video:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()

        if not ret:
            print('\033[31mКонец видео или ошибка чтения кадра...\033[0m')
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = model.track(frame, iou=0.5, conf=0.3, persist=True, imgsz=608, verbose=False)

        if results[0].boxes.id != None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            keypoints = results[0].keypoints.xy.cpu().numpy()
            for i, (box, id) in enumerate(zip(boxes, ids)):
                random.seed(int(id))
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                
                person_img = gray_frame[box[1]:box[3], box[0]:box[2]].copy()

                face_data = face_detected(person_img)

                # cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3],), color, 2)
                # cv2.putText(frame, f"Id {id}", (box[0], box[1]),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2,)
                
                if i < len(keypoints):
                    person_keypoints = keypoints[i]
                    frame = draw_keypoints(frame, person_keypoints, color)

                if face_data is not None:
                    face_coors, face_landmarks = face_data
                    for (x, y, w, h) in face_coors:
                        cv2.rectangle(frame,
                                        (box[0] + x, box[1] + y),
                                        (box[0] + x + w, box[1] + y + h),
                                        (0, 255, 0), 1)
                    if is_smiling(face_landmarks[0]):
                        cv2.putText(frame, 'Smile', (box[0] + x, box[1] + y + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 225, 0), 2)
                    

        if save_video:
            out.write(frame)
        if show_video:
            cv2.imshow('Frame', frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    if save_video:
        out.release()
    cv2.destroyAllWindows()
    
video_path_in = 'lesson22/src/in/video_1.mp4'
video_path_out = 'lesson22/src/out/video_1.mp4'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('lesson22/models/yolov8n-pose.pt')
model.to(device)

video_tracking(model, 0)