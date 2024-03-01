# from ultralytics.yolov8s.v8.predict import DetectionPredictor
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *



tracker = Tracker()
model=YOLO('yolov8s.pt')


cap=cv2.VideoCapture('kredpeoplecount4.mp4')
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

results = model.predict(source='rtsp://admin:Admin@123@192.168.1.137:554/1',show=True)
desired_width = 640
desired_height = 480

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame. Exiting...")
        break
    frame=cv2.resize(frame,(1020,500))
    results = model.predict(frame)
    frame_with_predictions = results.show()
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    personlistcount = []
    cv2.imshow("Frame with Predictions", frame_with_predictions)
    if cv2.waitKey(0) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()