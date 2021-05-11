import numpy as np
import os
import cv2 as cv
import tkinter as tk
from tkinter import *

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
count=0 # for stoping testing part in stop function
window=tk.Tk()
window.title("Face Recognition System")
window.geometry('1000x500')

uname=tk.StringVar() # global var to get dir name from entry widget
widgets=[] # will append widgets to remove them from screen

def sampling(path,username):
    cam =cv.VideoCapture(0,cv.CAP_DSHOW)
    count=0
    while cam.isOpened():
        
        x,frame = cam.read()
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.1,4)
        face=[]
        if len(faces) !=0:
            for(x,y,w,h) in faces:
                face=frame[y:y+h,x:x+w]#capuring face only
                cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                cv.imshow('img',frame)
            #print(face)
            face=np.asarray(face,dtype=np.uint8)
            try :
                f = cv.resize(face, (100,100), interpolation=cv.INTER_LINEAR)
                f = cv.cvtColor(f,cv.COLOR_BGR2GRAY)
                cv.imwrite(path+username.split(".")[-1]+"."+str(count)+'.jpg',f)
            except Exception as e:
                print(str(e))
                
            k =cv.waitKey(10)
            count=count+1
            if count== 100:
                cam.release()
                cv.destroyAllWindows()
                msg=Label(window,text='sample is taken').pack()

                m1=Label(window,text=" ").pack()
                
                widgets.append(msg)
                widgets.append(m1)
                   
               # print(len(face))
                break
            
    cam.release()
    cv.destroyAllWindows()

    
def training():
    path="./"
    train = []
    label = []
    dirc = os.listdir(path)
    for i in dirc:
        
        if os.path.isdir(path+i):
            onlyfiles = [f for f in os.listdir(path+i) if os.path.isfile(os.path.join(path+i,f))]#like k:/python/kaushal/kaushal99.jpg is file than add to list
            #print(onlyfiles)
            for files in onlyfiles:
                ipath = path+i+"/"+files
                #print(ipath)
                img = cv.imread(ipath,cv.IMREAD_GRAYSCALE)
                train.append(np.asarray(img,dtype=np.uint8))
                label.append(int(i.split(".")[-1]))
            #print(label)
            
            
        
    label=np.asarray(label,dtype=np.int32)
    return train ,label

def regi():
    un=uname.get()
    if(un!=""):
        path = "./"+un+"/"
        if os.path.isdir(path) and os.listdir(path)!=0:
            msg=Label(window,text="Enter another name your name is registerd ").pack()
            widgets.append(msg)
            
        else:
            if os.path.exists(path):
                if os.listdir(path)==0:
                    os.rmdir(path)
                
            os.mkdir(path)
            msg=Label(window,text="your sample will be taken ").pack()
            widgets.append(msg)
           # print(un)
            sampling(path,un)
            t ,l = training()
            model = cv.face.LBPHFaceRecognizer_create()
            model.train(t,l)
            model.save("./trainingdata.yml")
            
            print("trained")
    else:
        msg=Label(window,text="Enter your name Correctly ").pack()


def getdata():
    msg=Label(window,text="Enter your name in format name.number like Kaushal.123 ").pack()
    f_name = tk.Entry(window,textvariable = uname, font=('calibre',10,'normal')).pack()
    submit=Button(window,text='submit',command=regi).pack()
    widgets.append(msg)
    widgets.append(f_name)
    widgets.append(submit)
    
def stop():
    count=200
    
def test():
    model = cv.face.LBPHFaceRecognizer_create()
    model.read("./trainingdata.yml")
    cap =cv.VideoCapture(0)
    count=0
    dic={}
    while cap.isOpened():
        for i in os.listdir("./"):
            j=i.split(".")
            dic[j[-1]]=j[0]
            #name = j[0]
            # print(dic)
        count=count+1
        ret, img = cap.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)
        for (x,y,w,h) in faces:
            cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            idj,conf=model.predict(gray[y:y+h,x:x+w])
            #print(idj,conf)
            name="unknown"
            if str(idj) in dic.keys() and conf<80:
                name=dic[str(idj)]
                cv.putText(img,name,(y+100,400), cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                cv.imshow('img',img)
                #k =cv.waitKey(30)
                # to quit press enter
                
        if count == 200 or cv.waitKey(1)==13:
            #print(count)
            cap.release()
            cv.destroyAllWindows()
            break



label=Label(window,text='Welcome To Face Recognition System!',foreground="blue",font=("Arial", 25)).pack()

btn_test=Button(window,text='Recognize',command=test,fg="green").pack(padx=50,pady=50)
#btn_test_s=Button(window,text='Stop',command=stop,fg="green").pack(padx=50,pady=50)

btn_regi=Button(window,text='Register',command=getdata,fg="brown").pack()

Label(window,text="Project Created By  Kaushal Jani",foreground="blue",font=("Arial", 25)).pack(side=BOTTOM)
window.mainloop()

