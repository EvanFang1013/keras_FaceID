from sklearn import preprocessing #标准化数据模块
import numpy as np
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_classification 
from sklearn.svm import SVC 
import matplotlib.pyplot as plt
import os
import cv2
from sklearn import  metrics
import dlib



shape_predictor = dlib.shape_predictor('./dlib/shape_predictor_68_face_landmarks.dat') 
face_rec_model = dlib.face_recognition_model_v1('./dlib/dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()
from sklearn.externals import joblib


def svc_train(train_image,train_label):
    

    train_images, test_images, train_labels, test_labels = train_test_split(train_image, train_label, test_size=0.3,random_state=0)
#    test_images, _, test_labels, _ = train_test_split(train_image, train_label, test_size=0.7,random_state=0)
    print(len(train_labels))
    print(train_labels)
    print(len(test_labels))
   
    clf = SVC(C=2,gamma = 0.03,probability=True)
    clf.fit(train_images, train_labels)
    joblib.dump(clf, "./onepage.m")
    print("save sucessful")
    
#==============================================================================
#     test_y_predicted = clf.predict(test_images)
#     y_true = []
#     y_score = []
#     for i in range(len(test_y_predicted)):
#         score = clf.predict_proba(test_images[i].reshape(1,len(test_images[i])))
#         print(score)
#         y_score.append(score)
# #        if  test_y_predicted[i] ==  test_labels[i] and score>0.99:    
# #            y_true.append(1)
# #        else:
# #            y_true.append(0)
# #    y_true = np.array(y_true)
#    
#     print(y_score)
#     
# #    y_score = np.array(y_score)
#     
# #    y = np.linspace(0, 0.99, len(y_true))
# ##    y_score =clf.decision_function(test_images)
# ##    print(test_y_predicted)
# #  
# #    print(clf.score(test_images, test_labels))
# #    
# #    fpr, tpr, thresholds = metrics.roc_curve(y_true, y,pos_label=1)
# #
# #    
# #    auc = metrics.auc(fpr, tpr)
# #    print(auc)
#     
#     
#     
#     plt.figure()
#     lw = 2
#     plt.figure(figsize=(10,10))
#     plt.plot(fpr, tpr, color='darkorange',
#              lw=lw, label='ROC curve (area = %0.2f)' % auc) 
#     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic example')
#     plt.legend(loc="lower right")
#     plt.show()
# 
#     
# #    print(clf.predict(test_images[0].reshape(1,len(test_images[0]))))
#==============================================================================

    
def crossVal(train_image,train_label):
    
    from sklearn.model_selection import learning_curve,validation_curve
    
    
    param_range = np.logspace(-6, -2, 10)
    
    train_images, test_images, train_labels, test_labels = train_test_split(train_image, train_label, test_size=0,random_state=0)
    
#==============================================================================
#     train_sizes, train_scores, test_scores = learning_curve(
#     SVC(gamma=0.001,kernel='rbf',probability=True),train_images,train_labels, cv=10,
#     train_sizes=np.array([ 0.1, 0.33, 0.55, 0.78, 1. ]))
#     
# 
#     train_scores_mean = np.mean(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#              label="Training")
#     plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#             label="Cross-validation")
#     
#     plt.xlabel("Training examples")
#     plt.ylabel("scores")
#     plt.legend(loc="best")
#     plt.show()
#==============================================================================
    
    train_scores, test_scores = validation_curve(
    SVC(probability=True), 
    train_images, train_labels,param_name='gamma', param_range=param_range,
    cv=10,scoring='accuracy', n_jobs=1 )
    
    
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    #可视化图形
    plt.plot(param_range, train_scores_mean, 'o-', color="r",
             label="Training")
    plt.plot(param_range, test_scores_mean, 'o-', color="g",
            label="Cross-validation")
    
    plt.xlabel("gamma")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.show()
    
    
    


def data_process(image):
    
    image = cv2.resize(image,(100,100))
    rect = dlib.rectangle(0, 0, image.shape[1], image.shape[0])
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    
    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
#    RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    shape = shape_predictor(img_output, rect)

    face_descriptor = np.array(face_rec_model.compute_face_descriptor(img_output, shape))
    
    return face_descriptor
    
    
#==============================================================================
# def read_path(imgs, lbls, path_name, label):
#     
#     for dir_item in os.listdir(path_name):
# 
#         full_path = os.path.abspath(os.path.join(path_name, dir_item))
#         face = cv2.imread(full_path)
#         feature = data_process(face) 
#         feature_std = preprocessing.scale(feature)
#         
#         imgs.append(feature_std)
#         lbls.append(label)
#==============================================================================
images = []
paths = []       

def read_path(path_name): 
   
    for dir_item in os.listdir(path_name):
        
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        
        if os.path.isdir(full_path):    
            read_path(full_path)
        else:  
            
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)  
                
                feature = data_process(image)
                feature_std = preprocessing.scale(feature)
                images.append(feature_std)
                paths.append(path_name)
    labels = np.array([int(label.split('/')[-1]) for label in paths])  
                         
    imgs_std = np.array(images,dtype=np.float32)
    lbls_std = np.array(labels,dtype=np.uint8)                
    return imgs_std,lbls_std
    

    

if __name__ == "__main__":

     parent_dir = './data/good/'
     img_std ,lbl_std= read_path(parent_dir)
#     Val_dir = './data/Traindata/'
#     img_val ,lbl_val= read_path(Val_dir)
     
     


     
#     crossVal(img_std,lbl_std)
     
     svc_train(img_std,lbl_std)
    
    