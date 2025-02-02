from ultralytics import YOLO
import os
import numpy as np
from glob import glob

# file_test = open('testMONKEY.txt', 'a')
# file_val = open('valMONKEY.txt', 'a')
# file_train = open('trainMONKEY.txt', 'a')
# for image in glob('train_yolo/images/train/*tif'):
#     r = np.random.uniform(0, 1)
#     if r<0.8:
#         file_train.write(image+'\n')
#     elif r<0.9:
#         file_val.write(image+'\n')
#     else:
#         file_test.write(image+'\n')        
if __name__ == "__main__":
    model = YOLO("yolo11s.pt")
    results = model.train(
        data="monkey.yaml",
        epochs=2000,
        patience=100,
        imgsz=640,
        visualize=True,
        device=[0],
        optimizer='auto',  # default 'auto' SGD, Adam, AdamW, NAdam, RAdam, RMSProp
        # single_cls=True,
        # cos_lr=True,
        # lr0=0.01,  # default 0.01 SGD=1E-2, Adam=1E-3
        box=0,  # default 7.5
        cls=0.7,  # default 0.5
        label_smoothing=0.1,  # default 0.0
        hsv_h=0.015,
        hsv_s=0.3,  # default 0.3
        hsv_v=0.2,  # default 0.4
        degrees=67.0,  # default 0.0 -180.0-180.0
        translate=0.2,  # default 0.1
        scale=0.1,  # default 0.5
        # shear=60.0,  # default 0.0
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,
        # bgr=0.2,
        mosaic=1.0,
        mixup=0.5,  # default 0.0
        copy_paste=0.5,  # default 0.0
        copy_paste_mode='mixup',  # default ("flip", "mixup")
        auto_augment='randaugment',  # default (randaugment, autoaugment, augmix)
        erasing=0.4,
        crop_fraction=1.0)
