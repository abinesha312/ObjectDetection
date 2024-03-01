import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker2 import *
tracker2 = Tracker()


model=YOLO('yolov8s.pt')

area1=[(673,433),(787,418),(796,426),(678,447)]

area2=[(669,455),(836,424),(859,431),(668,478)]
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('kredpeoplecount4.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0


while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
#    frame=cv2.flip(frame,1)
    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    personlistcount = []
#    print(px)
    list=[]
             
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'person' in c:
           list.append([x1,y1,x2,y2])
           print(list)
        x1,y4,x4,y4,id=bbox
        cvresult = cv2.pointPolygonTest(np.array(area2,np.int32),((x2,y2)),False)
        if cvresult >=0:
            cv2.rectangle(frame,(x1,y3),(x4,y4),(0,255,0),2)
            cv2.circle(frame,(x4,y4),4,(255,0,255),-1)
            cv2.putText(frame,str(id),(x1,y3),cv2.FONT_HERSHEY_COMPLEX,(0.5),(255,255,255),1)
        
      
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(255,0,0),2)
    cv2.putText(frame,str('1'),(809,423),cv2.FONT_HERSHEY_COMPLEX,(0.5),(0,0,0),1)

    cv2.polylines(frame,[np.array(area2,np.int32)],True,(255,0,0),2)
    cv2.putText(frame,str('2'),(862,421),cv2.FONT_HERSHEY_COMPLEX,(0.5),(0,0,0),1)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()

