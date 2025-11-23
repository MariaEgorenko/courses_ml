import torch
import torchvision
from torchvision import transforms as T
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_frame_with_segmentation(model, frame):
    transform = T.Compose([
            T.Resize(520),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    input_tensor = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
        output_predictions = output.argmax(0) 

    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    r = Image.fromarray(output_predictions.byte().cpu().numpy())
    r = r.resize(frame.size, resample=Image.NEAREST)
    r.putpalette(colors)

    mask_rgb = r.convert("RGB")

    overlay = Image.blend(frame, mask_rgb, alpha=0.5)

    return overlay

def segmentation(model, source=0, is_video=True):

    if is_video:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print('\033[31mНе удалось получить доступ к камере или файлу...\033[0m')
            return
        while True:
            ret, frame = cap.read()
            if not ret:            
                print('\033[31mНе удалось прочитать кадр...\033[0m')
                break
            
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            overlay = get_frame_with_segmentation(model, img)
            cv_overlay = cv2.cvtColor(np.array(overlay), cv2.COLOR_RGB2BGR)

            cv2.imshow('video', cv_overlay)

            key = cv2.waitKey(1) & 0xff
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        try:
            frame = Image.open(source).convert('RGB')
        except FileNotFoundError:
            print(f"Файл {source} не найден.")
            return

        overlay = get_frame_with_segmentation(model, frame)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(frame)
        plt.title("Original")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(overlay) 
        plt.title("Overlay (Blend)")
        plt.axis('off')

        plt.show()
    
model = torchvision.models.segmentation.deeplabv3_resnet101(weights='DEFAULT')
model.to(device)
model.eval()

image_path = 'lesson23/src/images/bus.jpg'
video_path = 'lesson23/src/video/bus.mp4'

segmentation(model, video_path)
