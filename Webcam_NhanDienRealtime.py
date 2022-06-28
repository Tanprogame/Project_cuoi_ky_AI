import cv2
from keras.models import  load_model
import io
import numpy as np

#import model
model1 = load_model('model/gender_model.h5')
model2 = load_model('model/ages_model.h5')
model3 = load_model('model/emotion_model.h5')
face_detect = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

classes1 = ['Female','Male']
classes2 = ['1-5','6-17','18-30','31-55','56-75','76-100']
classes3 = ['Angry','Happy','Sad','Surprise']



#kết nối webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    #-- Detect faces

    faces = face_detect.detectMultiScale(frame)
    for (x,y,w,h) in faces:
        fm = frame[y:y+h,x:x+w]
        frame_1 = cv2.resize(fm, dsize=(100,100))
        frame_2 = cv2.resize(fm, dsize=(128,128))
        frame_3 = cv2.resize(fm, dsize=(224,224))
        tensor1 = np.expand_dims(frame_1, axis=0)
        tensor2 = np.expand_dims(frame_2, axis=0)
        tensor3 = np.expand_dims(frame_3, axis=0)

        #dự đoán giới tính
        pred1 = model1.predict(tensor1)
        class_id1 = np.argmax(pred1)
        class_name1 = classes1[class_id1]

        #dự đoán độ tuổi
        pred2 = model2.predict(tensor2)
        class_id2 = np.argmax(pred2)
        class_name2 = classes2[class_id2]

        #dự đoán cảm xúc
        pred3 = model3.predict(tensor3)
        class_id3 = np.argmax(pred3)
        class_name3 = classes3[class_id3]

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame, 'Gender: '+class_name1, (x,y-50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
        cv2.putText(frame, 'Age: '+class_name2, (x,y-30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255),2)
        cv2.putText(frame, 'Emotion: '+class_name3, (x,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),2)
    cv2.imshow('Capture - Face detection', frame)

    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()


