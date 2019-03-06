# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:35:37 2019

@author: nvidia
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 09:07:37 2019

@author: nvidia
"""

import cv2
import time
from detector import FaceDetector
import threading as td  
import queue
import numpy as np
from keras.models import load_model
from keras import backend as K
K.clear_session()
#from face_CNN_keras import Model
from data_process import IMAGE_SIZE,resize_image
   
        
def camera(qtask):
   # Frame = []
 
    qt = qtask
   
    cap = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480,format=(string)I420, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
    

    while cap.isOpened():
        # Capture frame-by-frame
         _, Frame = cap.read()
         if Frame.shape[0] == 0:
            break
         if qt.empty():
            qt.put(Frame)

         if cv2.waitKey(1) & 0xFF == ord('q'):
             break


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



        
class detectface():
    def __init__(self,thresh):
           
        face_detector = FaceDetector()
        
        self.q = queue.Queue(2)
        cap=td.Thread(target=camera,args=(self.q,))
        cap.start()
        time.sleep(1)
     
        now = time.time()
        
        self.model = load_model('./model/model03052.h5')      
        frame = []
        self.imgs = []
        self.name = ['Fang','GG','Unknow']
        
     
        print("[INFO]****************face detect start***********************")
        
        while not self.q.empty():
#            now = time.time()
            
            frame = self.q.get()

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
           
           
            if thresh:
                bboxes = face_detector.predict(rgb_frame, thresh)
            else:
                bboxes = face_detector.predict(rgb_frame)

            ann_frame = self.annotate_image(frame, bboxes)
            

            cv2.imshow('window', ann_frame)
            #print("FPS: {:0.2f}".format(1 / (time.time() - now)), end="\r", flush=True)
           # time.sleep(0.05)


         
    def annotate_image(self,frame, bboxes):
            ret = frame[:]
        
            img_h, img_w,_ = frame.shape
        
            for x, y, w, h, p in bboxes:
                
                images = ret[int(y - 3*h/4):int(y + 2*h/3),int(x - 2*w/3):int(x + 2*w/3)]
                             
                                        
                label,prob = self.face_predict(images)
                
               
                if prob >=0.95:
                    cv2.putText(ret,self.name[int(label)],(int(x-2*w/3),int(y-2*h/3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                else:
                    pass
                    
                print(label,prob)
                cv2.rectangle(ret, (int(x - 2*w/3), int(y - 3*h/4)), (int(x + 2*w/3), int(y + 2*h/3)), (0, 0, 0), 2)
                
            return ret
            
    
    def face_predict(self,image):

        
        image = resize_image(image)
        image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))        
        image = image.astype('uint8')
        image = image / 255

        result = self.model.predict_classes(image)
        result_pro = self.model.predict(image)
        result_pro = np.array(result_pro).astype(np.float32)
        pmax =result_pro.max()
        pmax = round(pmax.tolist(),4)
       
        
        return result[0],pmax
                
                

 
       
if __name__ == "__main__":
    thresh = 0.98
    df = detectface(thresh)
    
   

    
    
    
