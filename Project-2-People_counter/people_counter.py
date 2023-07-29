import cvzone
import math
import cv2 as cv
from ultralytics import YOLO
import numpy as np
from sort import *

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

model = YOLO('../yolov8m.pt')


cap = cv.VideoCapture('../Videos/people.mp4')
mask = cv.imread('mask2.png')
tracker = Sort(max_age= 50)

up_line = [120,293,420,293]
down_line = [420,293,730,293]

up_counter = []
down_counter = []

while True:
    success, img = cap.read()
    if not success:
        break
    graphics = cv.imread('graphics.png',cv.IMREAD_UNCHANGED)
    mask = cv.resize(mask,(img.shape[1],img.shape[0]),interpolation= cv.INTER_CUBIC)
    cropped_img = cv.bitwise_and(img,mask)

    results = model(cropped_img,stream= True)
    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil(box.conf[0]*100)/100
            cls = classNames[int(box.cls[0])]
            if cls == 'person' and conf > 0.3:
                detections = np.vstack((detections,np.array((x1,y1,x2,y2,conf))))

    cv.line(img,(up_line[0],up_line[1]),(up_line[2],up_line[3]),color=(0,0,255),thickness=2)
    cv.line(img,(down_line[0],down_line[1]),(down_line[2],down_line[3]),color=(0,0,255),thickness=2)

    results_tracker = tracker.update(detections)
    for r in results_tracker:
        x1, y1, x2, y2, id = r
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cp = (x1+(x2-x1)//2 , y1+ (y2-y1)//2)
        if up_line[0] < cp[0] < up_line[2] and up_line[1]-20 < cp[1] < up_line[3]+20:
            if id not in up_counter:
                up_counter.append(id)
                cv.line(img, (up_line[0], up_line[1]), (up_line[2], up_line[3]), color=(0, 255, 0), thickness=2)
        elif down_line[0] < cp[0] < down_line[2] and down_line[1]-20 < cp[1] < down_line[3] + 20:
            if id not in down_counter:
                down_counter.append(id)
                cv.line(img, (down_line[0], down_line[1]), (down_line[2], down_line[3]), color=(0, 255, 0), thickness=2)

        cvzone.cornerRect(img,(x1,y1,(x2-x1),(y2-y1)),8,colorR=(255,0,0),colorC=(0,0,255))
        cvzone.putTextRect(img,f'{id}',(max(0,x1),max(35,y1-20)),scale=0.8,thickness=1)

    img = cvzone.overlayPNG(img,graphics,(730,260))
    cv.putText(img,f'{len(up_counter)}',(935,350),fontFace= cv.FONT_HERSHEY_PLAIN,fontScale=5,color=(50,50,50),thickness=2)
    cv.putText(img,f'{len(down_counter)}',(1175,350),fontFace= cv.FONT_HERSHEY_PLAIN,fontScale=5,color=(50,50,50),thickness=2)


    cv.imshow('Window',img)
    cv.waitKey(1)

cv.destroyAllWindows()
print(f'Total People moved upwards: {len(up_counter)}')
print(f'Total People moved downwards: {len(down_counter)}')