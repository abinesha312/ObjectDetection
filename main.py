import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *

tracker = Tracker()
model = YOLO('yolov8s.pt')
area1 = [(496, 421), (704, 436), (704, 455), (483, 423)]
area2 = [(466, 421), (710, 452), (704, 461), (448, 423)]
door_region = [(600, 421), (650, 421), (650, 461), (600, 461)]  # Adjust the coordinates for your specific door location

people_count = 0

def is_inside_door_region(x, y):
    # Check if a point (x, y) is inside the door region
    return cv2.pointPolygonTest(np.array(door_region, np.int32), (x, y), False) >= 0

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('rtsp://admin:Admin@123@192.168.1.137:554/1')
# cap = cv2.VideoCapture('kredpeoplecount4.mp4')  # 0 for default camera, change if using a different camera

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))
    
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'person' in c:
            list.append([x1, y1, x2, y2])
    
    bbox_id = tracker.update(list)
    
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        if is_inside_door_region((x3 + x4) // 2, (y3 + y4) // 2):
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
            cv2.circle(frame, ((x3 + x4) // 2, (y3 + y4) // 2), 4, (255, 0, 255), -1)
            cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            if id not in tracker.entered_ids:  # Check if person has not already entered
                people_count += 1
                tracker.entered_ids.add(id)
                print(f"People count: {people_count}")

    cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, str('1'), (809, 423), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, str('2'), (862, 421), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    cv2.polylines(frame, [np.array(door_region, np.int32)], True, (0, 255, 0), 2)
    cv2.putText(frame, str('Door'), ((door_region[0][0] + door_region[1][0]) // 2, (door_region[0][1] + door_region[2][1]) // 2),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow("RGB", frame)
    
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()
