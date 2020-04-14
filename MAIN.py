
import sklearn
import numpy as np
import os
import cv2 as cv
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')


def sampling(path,uname):
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
                cv.imwrite(path+uname.split(".")[-1]+"."+str(count)+'.jpg',f)
            except Exception as e:
                print(str(e))
                
            k =cv.waitKey(10)
            count=count+1
            if count== 100:
                cam.release()
                cv.destroyAllWindows()
                print('sample is taken')
               # print(len(face))
                break
            
    cam.release()
    cv.destroyAllWindows()
def training():
    path="./"
    train = []
    label = []
    dir = os.listdir(path)
    for i in dir:
        
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
        
    
p=1
print()
print()
print("\t\t\t\tWelcome to face detection system!!!\t\t\t")
print()
print()
while p>0:
    try :
        
        c=int(input(" To register new user enter 1\n To predict enter 2\n To test an image press 3 "))
        if c==1:
            uname=input("enter your name in format name.number ")
            path = "./"+uname+"/"
            if os.path.isdir(path) and os.listdir(path)!=0  :
                
                print("enter another name your name is registerd ")
            
            else:
                if os.path.exists(path):
                    if os.listdir(path)==0:
                        os.rmdir(path)
                
                os.mkdir(path)
                
                print("your sample will be taken ")
                sampling(path,uname)
                t ,l = training()
                model = cv.face.LBPHFaceRecognizer_create()
                model.train(t,l)
                model.save("./trainingdata.yml")
        if c==2:
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
                    if str(idj) in dic.keys() and conf<65:
                        name=dic[str(idj)]
                    
                    cv.putText(img,name,(y+100,400), cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                    cv.imshow('img',img)
                #k =cv.waitKey(30)
                # to quit press enter
                
                if count == 200 or cv.waitKey(1)== 13 :
                    print(count)
                    cap.release()
                    cv.destroyAllWindows()
                    break
            #cap.release()
            #cv.destroyAllWindows()
        if c==3:
            dic={}
            for i in os.listdir("./"):
                j=i.split(".")
                dic[j[-1]]=j[0]
            path=input(" enter path of image ")
            img1 = cv.imread(path)
            img = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(img,1.1,4)
            model = cv.face.LBPHFaceRecognizer_create()
            model.read("./trainingdata.yml")
            if len(faces) !=0:
                #print("1")
                for(x,y,w,h) in faces:
                    face=img[y:y+h,x:x+w]#capuring face only
                    cv.rectangle(img1,(x,y),(x+w,y+h),(255,0,0),2)
                    idj,conf=model.predict(img[y:y+h,x:x+w])
                    #print(idj,conf)
                    name="unknown"
                    #cv.imshow('img',img1)
                    h = img.shape[0]
                    w = img.shape[1]
                    ck=10
                    ck+=10
                    if str(idj) in dic.keys():
                        if conf<55:
                            name=dic[str(idj)]
                            
                        else :
                            name="unknown"
                    cv.putText(img1,name,(h,100+ck), cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
                    cv.imshow('img',img1)
                    cv.waitKey(100)
                    cv.imwrite("./recognized.jpg",img1)
                
            
    except Exception as e:
        pass
    print(" To continue press 1\n To stop press 0")
    try:
       p=int(input())
    except Exception as e:
        print("enter correct number")
    
print("\t\t\t\tThank YOU!!!\t\t\t")
        
    
    

    
        
        
    
    
