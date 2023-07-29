import math

import numpy as np
from ultralytics import YOLO
import cv2 as cv
import cvzone
from sort import *

cap = cv.VideoCapture('../Videos/cars.mp4')

model = YOLO('../yolov8m.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
mask = cv.imread('mask.png')
mask = cv.resize(mask,(1280,720),interpolation=cv.INTER_CUBIC)

tracker = Sort(max_age= 50)
line = [420,293,730,293]
total_cars = []
while True:
    success, img = cap.read()
    if not success:
        break
    cropped_image = cv.bitwise_and(img,mask)
    graphics = cv.imread('graphics.png',cv.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img,graphics,(0,0))
    results = model(cropped_image,stream=True)
    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2= int(x1),int(y1),int(x2),int(y2)
            cvzone.cornerRect(img,(x1,y1,(x2-x1),(y2-y1)),l=8)

            conf = math.ceil(box.conf[0]*100)/100
            cls = classNames[int(box.cls[0])]
            if cls in ['car','bus','truck','motorbike'] and conf > 0.3:
                detections = np.vstack((detections,np.array([x1,y1,x2,y2,conf])))

    cv.line(img,(line[0],line[1]),(line[2],line[3]),color = (0,0,255),thickness=3)
    results_tracker = tracker.update(detections)

    for r in results_tracker:
        x1, y1, x2, y2, id = r
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        center = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)
        if line[0] < center[0] < line[2] and line[1]-20 < center[1] < line[3]+20:
            if id not in total_cars:
                total_cars.append(id)
            cv.line(img, (line[0], line[1]), (line[2], line[3]), color=(0, 255, 0), thickness=3)

        cvzone.cornerRect(img, (x1, y1, (x2 - x1), (y2 - y1)), l=8,colorR=(255,0,0))
        cvzone.putTextRect(img, f'{id}', (max(0, x1), max(35, y1 - 20)),
                           scale=1, thickness=1, offset=3)

    cv.putText(img,f' {len(total_cars)}',(235,100),cv.FONT_HERSHEY_PLAIN,fontScale=5, color=(50,50,50),thickness=5)
    cv.imshow('Camera',img)
    cv.waitKey(1)

cv.destroyAllWindows()

print(len(total_cars))



