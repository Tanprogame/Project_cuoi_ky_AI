from tkinter import *
import tkinter
from tkinter.font import BOLD
from tkinter.filedialog import Open
import cv2
from PIL import ImageTk, Image
from keras.models import  load_model
import io
import numpy as np

#tạo tkinter
App = Tk()
App.title('App nhận diện khuôn mặt Realtime')
App.geometry('1000x650')
App['bg'] = '#99FFFF'                   #background color

#khai bao bien
show = 0
dt = 0
reg = 0

#load model 
face_detect = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
model1 = load_model('model/gender_model.h5')
model2 = load_model('model/ages_model.h5')
model3 = load_model('model/emotion_model.h5')

classes1 = ['Female','Male']
classes2 = ['1-5','6-17','18-30','31-55','56-75','76-100']
classes3 = ['Angry','Happy','Sad','Surprise']

#kết nối webcam
cam = cv2.VideoCapture(0)
#canvas_w = cam.get(cv2.CAP_PROP_FRAME_WIDTH) // 2
#canvas_h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2

#tạo canvas
canvas = Canvas(App, width=600,height=400,bg='black')
canvas.place(x=20,y=180)

#tạo label 1
name1 = Label(App,  text = 'TRƯỜNG ĐẠI HỌC SƯ PHẠM KỸ THUẬT TP.HCM', font=('Time New Roman',18, BOLD),bg='#99FFFF')
name1.place(x = 250,y=10)

#tạo label 2
name2 = Label(App,  text = 'KHOA CƠ KHÍ CHẾ TẠO MÁY', font=('Time New Roman',18,BOLD),bg='#99FFFF')
name2.place(x = 350,y=40)

#tạo tên SVTH
sv = Label(App,  text = 'Project cuối kỳ\nMôn học: Trí tuệ nhân tạo\nGVHD: Nguyễn Trường Thịnh\nSVTH: Bùi Nhật Tấn\nMSSV: 19146385', 
            font=('Time New Roman',14,BOLD),bg='#CC66FF',justify='left')
sv.place(x = 650,y=200)

#tạo tên đề tài
detai = Label(App,  text = 'Đề tài: DỰ ĐOÁN GIỚI TÍNH, ĐỘ TUỔI, CẢM XÚC QUA ĐẶC ĐIỂM KHUÔN MẶT REALTIME', 
            font=('Time New Roman',16,BOLD),bg='#99FFFF',fg='red')
detai.place(x = 50,y=120)

#tạo label chứa logo UTE CKM
img1 = cv2.imread('image/ute.jfif')
img1 = cv2.resize(img1,(80,80))
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = cv2.imread('image/ckm.png')
img2 = cv2.resize(img2,(80,80))
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

img3 = cv2.imread('image/face.jpg')
img3 = cv2.resize(img3,(600,400))
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

ute = ImageTk.PhotoImage(image=Image.fromarray(img1))
ckm = ImageTk.PhotoImage(image=Image.fromarray(img2))
anh_nen = ImageTk.PhotoImage(image=Image.fromarray(img3))

logo1 = Label(App,image=ute)
logo1.place(x = 10,y=10)
logo2 = Label(App,image=ckm)
logo2.place(x = 100,y=10)

#tạo button bật/tắt webcam
def nhanvao():
    global show
    show = 1-show
    if show==0:
        btn1['text'] = 'Webcam off'
        btn1['bg'] = 'red'
    else:
        btn1['text'] = 'Webcam on'
        btn1['bg'] = 'green'
btn1 = Button(App, text = 'Webcam off',width=10,height=2, bg='red', font=('Time New Roman',14,BOLD), command=nhanvao, borderwidth=5)
btn1.place(x=650,y=400)

#tạo button detect khuôn mặt
def detect():
    global dt
    dt = 1-dt
    if dt==0:
        btn2['text'] = 'Detect off'
        btn2['bg'] = 'red'
    else:
        btn2['text'] = 'Detect on'
        btn2['bg'] = 'green'
btn2 = Button(App, text = 'Detect off',width=10,height=2, bg='red', font=('Time New Roman',14,BOLD), command=detect, borderwidth=5)
btn2.place(x=800,y=400)

#tạo button dự đoán giới tính, tuổi, cảm xúc
def recognition():
    global reg
    reg = 1-reg
    if reg==0:
        btn3['text'] = 'recognize off'
        btn3['bg'] = 'red'
    else:
        btn3['text'] = 'recognize on'
        btn3['bg'] = 'green'
btn3 = Button(App, text = 'recognize off',width=10,height=2, bg='red', font=('Time New Roman',14, BOLD), command=recognition, borderwidth=5)
btn3.place(x=650,y=480)

btn4 = Button(App, text = 'Exit',width=10,height=2, bg='red', font=('Time New Roman',14, BOLD), command=App.quit, borderwidth=5)
btn4.place(x=800,y=480)

#cập nhật hình ảnh lên canvas
def update_frame():
    global canvas, photo, bw, count, show, dt, reg
    # Doc tu camera
    if show == 1:
        ret, frame = cam.read()
        # Ressize
        frame = cv2.resize(frame, dsize=(600,400))
        # Chuyen he mau
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #detect khuôn mặt
        if dt ==1:
            faces = face_detect.detectMultiScale(frame)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                if reg ==1:
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

                    #Dự đoán độ tuổi
                    pred2 = model2.predict(tensor2)
                    class_id2 = np.argmax(pred2)
                    class_name2 = classes2[class_id2]

                    #Dự đoán cảm xúc
                    pred3 = model3.predict(tensor3)
                    class_id3 = np.argmax(pred3)
                    class_name3 = classes3[class_id3]

                    cv2.putText(frame, 'Gender: '+class_name1, (x,y-50),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),2)
                    cv2.putText(frame, 'Age: '+class_name2, (x,y-30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),2)
                    cv2.putText(frame, 'Emotion: '+class_name3, (x,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2)
        # Convert hanh image TK
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        
        # Show
        canvas.create_image(0,0, image = photo, anchor=tkinter.NW)
        
    else:
        canvas.create_image(0,0, image = anh_nen, anchor=tkinter.NW)
    App.after(15, update_frame)
    
    

update_frame()


App.mainloop()