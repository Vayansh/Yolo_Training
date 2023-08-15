import cv2 as cv
import cvzone
from ultralytics import YOLO
from poker_hand_detector_function import poker_hand_detector

img = cv.imread('two_pair.jpeg')

model = YOLO('playing_card_model2.pt')


nc= 52
class_names= ['10C', '10D', '10H', '10S',
              '2C', '2D', '2H', '2S',
              '3C', '3D', '3H', '3S',
              '4C', '4D', '4H', '4S',
              '5C', '5D', '5H', '5S',
              '6C', '6D', '6H', '6S',
              '7C', '7D', '7H', '7S',
              '8C', '8D', '8H', '8S',
              '9C', '9D', '9H', '9S',
              'AC', 'AD', 'AH', 'AS',
              'JC', 'JD', 'JH', 'JS',
              'KC', 'KD', 'KH', 'KS',
              'QC', 'QD', 'QH', 'QS']

results = model(img)
hand = []
for r in results:
    boxes = r.boxes
    for box in boxes:
        k = class_names[int(box.cls[0])]
        if k not in hand:
            hand.append(k)

# cam = cv.VideoCapture(0)
#
# while True:
#     success, frame = cam.read()
#     results = model(frame,stream= True)
#
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             k = class_names[int(box.cls[0])]
#             if k not in hand:
#                 hand.append(k)
#     if len(hand) == 5:
#         break
#     cv.imshow('win',frame)
#     cv.waitKey(1)

print(hand)
print(poker_hand_detector(hand))
