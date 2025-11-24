import torch, detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

import matplotlib
matplotlib.use('TkAgg')  # для отображения картинки через plt.imshow() в wsl ubuntu
import matplotlib.pyplot as plt

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)

def get_segmented_image(frame):

    outputs = predictor(frame)

    v = Visualizer(
        frame[:, :, ::-1], 
        MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), 
        scale=1.0, 
        instance_mode=ColorMode.SEGMENTATION 
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    return out.get_image()

def segmentation(source=0, is_video=True, show=True, save=False, output_path=''):

    if is_video:
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print('\033[31mНе удалось получить доступ к камере или файлу...\033[0m')
            return

        if save:
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps    = cap.get(cv2.CAP_PROP_FPS)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            if not output_path:
                output_path = 'output_video.mp4'
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:

            ret, frame = cap.read()
            if not ret:            
                print('\033[31mНе удалось прочитать кадр или закончилось видео...\033[0m')
                break

            rgb_res = get_segmented_image(frame)

            res = cv2.cvtColor(rgb_res, cv2.COLOR_RGB2BGR)

            if show:
                cv2.imshow('video', res)
                key = cv2.waitKey(1) & 0xff
                if key == 27:
                    break
            if save:
                out.write(res)
        
        cap.release()
        if save:
            out.release()
        cv2.destroyAllWindows()

    else:
        img = cv2.imread(source)
        res = get_segmented_image(img)
        if save:
            cv2.imwrite(output_path, res)
        if show:
            plt.figure(figsize=(12, 8))
            plt.imshow(res)
            plt.show()

# image_path = 'lesson23/src/images/bus.jpg'
# out_image_path = 'lesson23/src/images/out_bus.jpg'
video_path = 'lesson23/src/video/bus.mp4'
out_video_path = 'lesson23/src/video/output_bus.mp4'

segmentation(video_path, is_video=True, show=False, save=True, output_path=out_video_path)