import cv2
from ultralytics import YOLO
import torch
import random
import dlib

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
    print('детекция лиц')
    faces = detector(gray_frame)
    if len(faces) < 1:
        print('лица не найдены')
        return None
    print(f'количество найденных лиц: {len(faces)}')
    faces_coor = []
    faces_landmarks = []

    print('сохранение координат и лэндмарок')
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        landmarks = predictor(gray_frame, face)

        x = max(0, x)
        y = max(0, y)

        faces_coor.append((x, y, w, h))
        faces_landmarks.append(landmarks)

    return faces_coor, faces_landmarks

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
                
                print(f'координаты для вырезки изображения: {box[1]}, {box[3]}, {box[0]}, {box[2]}')
                print(f'размер исходного изображения: {gray_frame.shape}')
                person_img = gray_frame[box[1]:box[3], box[0]:box[2]]
                cv2.imshow('frame', person_img)
                key = cv2.waitKey(0)

                face_data = face_detected(person_img)

                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3],), color, 2)
                cv2.putText(frame, f"Id {id}", (box[0], box[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2,)
                
                if i < len(keypoints):
                    person_keypoints = keypoints[i]
                    frame = draw_keypoints(frame, person_keypoints, color)

                if face_data is not None:
                    print('рисование прямоугольника лица')
                    f_coors, f_landmarks = face_data
                    for (fx, fy, fw, fh) in f_coors:
                        cv2.rectangle(frame,
                                        (box[0] + fx, box[1] + fy),
                                        (box[0] + fx + fw, box[1] + fy + fh),
                                        (0, 255, 0), 1)
        if key == 27:
            break

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