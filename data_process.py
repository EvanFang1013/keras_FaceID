#data_process.py
"""
Created on Tue Feb 26 10:31:39 2019

@author: nvidia
"""

import os
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import pickle

IMAGE_SIZE = 100

def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    
#==============================================================================
#     top, bottom, left, right = (0, 0, 0, 0)
#     h, w, _ = image.shape
#     longest_edge = max(h, w)
#     if h < longest_edge:
#         dh = longest_edge - h
#         top = dh // 2
#         bottom = dh - top
#     elif w < longest_edge:
#         dw = longest_edge - w
#         left = dw // 2
#         right = dw - left
#     else:
#         pass
#     BLACK = [0, 0, 0]
#     constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
#     img = cv2.resize(constant, (height, width))
#==============================================================================
    
    img = cv2.resize(image,(height,width))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ= cv2.equalizeHist(gray)
#    img = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    _,binary = cv2.threshold(equ,80, 255, cv2.THRESH_BINARY) 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = cv2.split(hsv)[1]
    merged = cv2.merge([hsv,binary,gray])
    
#    cv2.imwrite('./10.jpg', merged) 
    
    return merged
    

def cluster(label,eps,n):
    
    codemodel = './codingdata/train_data.pickle'
    data = pickle.loads(open(codemodel,"rb").read())#args["encodings"], "rb").read())
    data = np.array(data)
    encodings = [d["encoding"] for d in data if d["label"]==label]
    path = [d["imagePath"] for d in data if d["label"]==label]
    
    # cluster the embeddings
    print("[INFO] clustering...")
    clt = DBSCAN(eps=eps,min_samples=n,n_jobs=-1)#args["jobs"])#(metric="euclidean", n_jobs=args["jobs"])
    clt.fit(encodings)
    label_all = clt.labels_
    # determine the total number of unique faces found in the dataset

    label_index=label
    goodindex = np.where(label_all==0)[0]
    badindex = np.where(label_all==-1)[0]
    print(label_index,'good: %s'%len(goodindex),"bad: %s"%len(badindex))
#    print(goodindex)
    
    return path,goodindex,badindex,label_index   
    
def train_data(train_images,train_labels,path,goodindex,badindex,label_index):
      
    Newgoodindex=np.random.choice(goodindex,60) 
    Newbadindex = np.random.choice(badindex,200)

    for index in Newgoodindex:
        image = cv2.imread(path[index])
        image = resize_image(image)             
        train_images.append(image)
        train_labels.append(int(label_index))
    for index in Newbadindex:
        image = cv2.imread(path[index])
        image = resize_image(image)    
        train_images.append(image)
        train_labels.append(int(label_index))
        
def Val_data():
    Val_img= []
    Val_label = []
    codemodel = './codingdata/Val_data.pickle'
    data = pickle.loads(open(codemodel,"rb").read())
    data = np.array(data)
    for i in range(2):       
        path = [d["imagePath"] for d in data if d["label"]==str(i)]
        for img in path:
            image = cv2.imread(img)   
            image = resize_image(image)
            Val_img.append(image)
            Val_label.append(i)
    
    
#    read_path(Val_img, Val_label,'./data/Valdata/2', 2)
    
    Val_img = np.array(Val_img)
    Val_label = np.array(Val_label)
            
    return Val_img,Val_label
        
    
    
    
def loaddata():
    
    train_images = []
    train_labels = []    
#    parent_dir = './data/Traindata'
    path,goodindex,badindex,label_index = cluster(label='0',eps=0.18,n=16)
    train_data(train_images,train_labels,path,goodindex,badindex,label_index)    
    path,goodindex,badindex,label_index = cluster(label='1',eps=0.16,n=16)
    train_data(train_images,train_labels,path,goodindex,badindex,label_index) 
#==============================================================================
#     path,goodindex,badindex,label_index = cluster(label='0',eps=0.2,n=6)
#     train_data(train_images,train_labels,path,goodindex,badindex,label_index)    
#     path,goodindex,badindex,label_index = cluster(label='1',eps=0.19,n=10)
#     train_data(train_images,train_labels,path,goodindex,badindex,label_index) 
#     path,goodindex,badindex,label_index = cluster(label='2',eps=0.19,n=7)
#     train_data(train_images,train_labels,path,goodindex,badindex,label_index)
#     path,goodindex,badindex,label_index = cluster(label='3',eps=0.23,n=8)
#     train_data(train_images,train_labels,path,goodindex,badindex,label_index)
#==============================================================================
#    read_path(train_images, train_labels, parent_dir+"/2", 2)
    
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    
    return train_images,train_labels
    
    
    
def read_path(imgs, lbls, path_name, label):
    for dir_item in os.listdir(path_name):

        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        face = cv2.imread(full_path)
        face = resize_image(face)  
        
        imgs.append(face)
        lbls.append(label)
        
   

    

    
if __name__ == '__main__':
        a,b = Val_data()
        c,d = loaddata()
        print(d)
       
        