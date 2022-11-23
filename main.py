import face_recognition
import csv
from datetime import datetime
import cv2
import os
import numpy as np
video_capture=cv2.VideoCapture(0)
encoding_data_list=[]
name_list=[]
path=os.getcwd()+"/photos"
for image in os.listdir(path):
    if image.endswith(".jpg") or image.endswith(".jpg") or image.endswith(".jpg"):
        ei=face_recognition.load_image_file("photos/"+image)
        ed=face_recognition.face_encodings(ei)
        encoding_data_list.append(ed)
        name=os.path.splitext(image)[0]
        name_list.append(name)
students=name_list.copy()
face_locations=[]
face_encoding=[]
face_name=[]
s=True

now=datetime.now()
current_date=now.strftime("%Y-%m-%d")

f=open(current_date+".csv",'w+',newline='')
lnwriter=csv.writer(f)

while True:
    _,frame=video_capture.read()
    small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame=small_frame[:,:,::-1]
    if s:
        face_locations=face_recognition.face_locations(rgb_small_frame)
        face_encodings=face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names=[]
        for face_encoding in face_encodings:
            matches=face_recognition.compare_faces(encoding_data_list,face_encoding)
            name=""
            face_distance = face_recognition.face_distance(encoding_data_list,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = name_list[best_match_index]
            face_names.append(name)
            if name in name_list:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time=now.strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])
    cv2.imshow("attendence system",frame)
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
f.close()
