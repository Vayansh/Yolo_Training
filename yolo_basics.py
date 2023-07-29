from ultralytics import YOLO
import cv2 as cv

model = YOLO('yolov8m.pt')
results = model('Images/3.png',show = True)
cv.waitKey(0)