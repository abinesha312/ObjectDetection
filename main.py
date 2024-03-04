import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*
import cvzone
import numpy as np
import requests
from datetime import datetime, timezone


AddCount_url = "https://gc86e6ffd3f8773-db87cxy.adb.us-ashburn-1.oraclecloudapps.com/ords/kred/KredVision/AddCount/"
current_time = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

model=YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
# cap=cv2.VideoCapture(0)
# cap=cv2.VideoCapture('busfinal.mp4')
cap = cv2.VideoCapture('rtsp://admin:Admin@123@68.185.201.179:554/1')
# cap = cv2.VideoCapture('rtsp://admin:Admin@123@192.168.1.137:554/1')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0
area1=[(259,488),(281,499),(371,499),(303,466)]
tracker=Tracker()
going_in={}
counter=[]
while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
    results=model.predict(frame)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    list=[]
    for index,row in px.iterrows(): 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'person' in c:
            list.append([x1,y1,x2,y2])
    bbox_id = tracker.update(list)
    print(bbox_id)
    for id,rect in bbox_id.items():
        x3,y3,x4,y4 = rect
        cx=x3
        cy=y4
        results=cv2.pointPolygonTest(np.array(area1,np.int32),((cx,cy)),False)
        if results>=0:
            cv2.circle(frame,(cx,cy),6,(0,255,0),-1)
            cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,255),2)
            cvzone.putTextRect(frame,f'{id}',(x3,y3),1,1)
            if counter.count(id)==0:
                counter.append(id)

    cv2.polylines(frame,[np.array(area1,np.int32)],True,(255,0,0),2)
    p=len(counter)
    payload = {"gender": "Male","count": p,"entrytime": current_time,"created_date": current_time,"created_by": "User2"}
    response = requests.post(AddCount_url, json=payload)
    cvzone.putTextRect(frame,f'Counter:-{p}',(50,60),2,2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
