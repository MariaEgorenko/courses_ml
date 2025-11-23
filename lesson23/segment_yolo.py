from ultralytics import YOLO
import torch
import cv2

def segmentation(model, source=0):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print('\033[31mНе удалось получить доступ к камере...\033[0m')
        exit()

    while True:

        ret, frame = cap.read()

        if not ret:
            print('\033[31mНе удалось прочитать кадр...\033[0m')
            break

        results = model(frame)

        result = results[0]
        res_img = result.plot()

        cv2.imshow('video', res_img)

        key = cv2.waitKey(1) & 0xff
        if key == 27:
            break
        
    cap.release()
    cv2.destroyAllWindows()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = YOLO('lesson23/models/yolo11n-seg.pt')
model.to(device)

medai_path = 'lesson23/src/video/bus.mp4'

segmentation(model, medai_path)