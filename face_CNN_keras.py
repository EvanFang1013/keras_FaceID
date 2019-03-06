#face_CNN_keras.py


import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Flatten , AveragePooling2D
#from keras.layers import Convolution2D, MaxPooling2D
#from keras.layers import Conv2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
import matplotlib.pyplot as plt

from keras.optimizers import Adam
#from pyloaddata import loaddata, load_tdata
#from picturepraction import resize_image, IMAGE_SIZE
from data_process import loaddata,Val_data,IMAGE_SIZE,resize_image


class Dataset:
    def __int__(self):

       
        self.train_images = None
        self.train_labels = None

        self.valid_images = None
        self.valid_labels = None

        self.test_images = None
        self.test_labels = None

        self.input_shape = None

 
    def load(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE, img_channels=3, nb_classes=3):
        train_image,train_label = loaddata()
        timg , tlabel = Val_data()

        train_images, _, train_labels, _ = train_test_split(train_image, train_label, test_size=0.2,random_state=0)
        test_images, valid_images, test_labels, valid_labels = train_test_split(timg, tlabel, test_size=0.6,random_state=0)
       
     
        if (K.image_dim_ordering() == 'th'):
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)
        else:                              
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)

    
        print(train_images.shape[0], 'train samples')
        print(valid_images.shape[0], 'valid_samples')
        print(test_images.shape[0], 'test_samples')

     
        train_labels = np_utils.to_categorical(train_labels, nb_classes)
        valid_labels = np_utils.to_categorical(valid_labels, nb_classes)
        test_labels = np_utils.to_categorical(test_labels, nb_classes)

        train_images = train_images.astype('uint8')
        valid_images = valid_images.astype('uint8')
        test_images = test_images.astype('uint8')
        train_images = train_images /255
        valid_images =valid_images /255
        test_images = test_images /255

        self.train_images = train_images
        self.valid_images = valid_images
        self.test_images = test_images
        self.train_labels = train_labels
        self.valid_labels = valid_labels
        self.test_labels = test_labels

     


class Model():
    def __init__(self):
        self.model = None

    def build_model(self, dataset, nb_classes=3):
#==============================================================================
#         from keras.applications.xception import Xception
#         self.model = Xception(include_top=True,weights=None,input_tensor=None, input_shape=(100,100,3), pooling='max', classes=2)
#==============================================================================
        
#------------------- Model -----------------------#
#==============================================================================
#         self.model = Sequential() 
#         self.model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=dataset.input_shape, activation='relu'))
#         self.model.add(MaxPooling2D(pool_size=(2, 2)))                         
#         self.model.add(Dropout(0.5)) 
#         self.model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
#         self.model.add(Dropout(0.5))
#         self.model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
#         self.model.add(MaxPooling2D(pool_size=(2, 2)))
#         self.model.add(Dropout(0.5))
#         self.model.add(Flatten())   
#         self.model.add(Dense(512))
#         self.model.add(Activation('relu'))
#         self.model.add(Dropout(0.5))
#         self.model.add(Dense(nb_classes))
#         self.model.add(Activation('softmax'))
#         self.model.summary()
#         
#==============================================================================
        
#-------------------ALexnet Model -----------------------#       
#==============================================================================
#         self.model = Sequential()
#         self.model.add(Conv2D(96,(11,11),strides=(4,4),input_shape=dataset.input_shape,padding='valid',activation='relu',kernel_initializer='uniform'))
#         self.model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
#         self.model.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
#         self.model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
#         self.model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
#         self.model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
#         self.model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
#         self.model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
#         self.model.add(Flatten())
#         self.model.add(Dense(4096,activation='relu'))
#         self.model.add(Dropout(0.5))
#         self.model.add(Dense(4096,activation='relu'))
#         self.model.add(Dropout(0.25))
# 
#         self.model.add(Dense(nb_classes))
# 
#         self.model.add(Activation('softmax'))
#         self.model.summary()
#         
#==============================================================================
        from keras.layers import Dense,  Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
        
        self.model = Sequential()

        # 1 - Convolution
        self.model.add(Conv2D(64,(3,3), padding='valid', input_shape=dataset.input_shape))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
        
        # 2nd Convolution layer
        self.model.add(Conv2D(128,(5,5), padding='valid'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
        
        # 3rd Convolution layer
        self.model.add(Conv2D(512,(3,3), padding='valid'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
        
        # 4th Convolution layer
        self.model.add(Conv2D(512,(3,3), padding='valid'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
        
        # Flattening
        self.model.add(Flatten())
        
        # Fully connected layer 1st layer
        self.model.add(Dense(256))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        
        # Fully connected layer 2nd layer
        self.model.add(Dense(512))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        
        self.model.add(Dense(nb_classes))
        self.model.add(Activation('softmax'))
        self.model.summary()
#==============================================================================
#         self.model = Sequential()
#  
#         #1st convolution layer
#         self.model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(100,100,3)))
#         self.model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))
#          
#         #2nd convolution layer
#         self.model.add(Conv2D(64, (3, 3), activation='relu'))
#         self.model.add(Conv2D(64, (3, 3), activation='relu'))
#         self.model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))
#          
#         #3rd convolution layer
#         self.model.add(Conv2D(128, (3, 3), activation='relu'))
#         self.model.add(Conv2D(128, (3, 3), activation='relu'))
#         self.model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))
#          
#         self.model.add(Flatten())
#          
#         #fully connected neural networks
#         self.model.add(Dense(1024, activation='relu'))
#         self.model.add(Dropout(0.2))
#         self.model.add(Dense(1024, activation='relu'))
#         self.model.add(Dropout(0.2))
#          
#         self.model.add(Dense(nb_classes, activation='softmax'))
#         self.model.summary()
#==============================================================================
        

    def train(self, dataset, batch_size = 20, nb_epoch = 50, data_augmentation = True):        
        sgd = SGD(lr = 0.0001, decay = 1e-6, momentum = 0.9, nesterov = True)    
        
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])
#==============================================================================
#         adam = Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-6)
#         self.model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
#==============================================================================

        if not data_augmentation:            
            self.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size = batch_size,
                           nb_epoch = nb_epoch,
                           validation_data = (dataset.valid_images, dataset.valid_labels),
                           shuffle = True)

        else:            
   
            datagen = ImageDataGenerator(
                featurewise_center = False,             
                samplewise_center  = False,             
                featurewise_std_normalization = False, 
                samplewise_std_normalization  = False,  
                zca_whitening = False,                  
                rotation_range = 10,                    
                width_shift_range  = False,               
                height_shift_range = False,              
                horizontal_flip = True,               
                vertical_flip = False)                 
 
            datagen.fit(dataset.train_images)                        

            train_history = self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,
                                                   batch_size = batch_size),
                                     samples_per_epoch = dataset.train_images.shape[0],
                                     nb_epoch = nb_epoch,
                                     validation_data = (dataset.valid_images, dataset.valid_labels))
                                     
                    
            self.save_model("./model/model03052.h5")                         
            self.plot_train_history(train_history,'acc', 'val_acc')
            self.plot_train_history(train_history, 'loss', 'val_loss')
            
    def plot_train_history(self,train_history, train, validation):  
        plt.plot(train_history.history[train])  
        plt.plot(train_history.history[validation])  
        plt.title('Train History')  
        plt.ylabel(train)  
        plt.xlabel('Epoch')  
        plt.legend(['train', 'validation'], loc='upper left')  
        plt.show() 
    
  
    def save_model(self, MODEL_PATH):
        self.model.save(MODEL_PATH)

    def load_model(self, MODEL_PATH):
        self.model = load_model(MODEL_PATH)

    def evaluate(self, dataset):
        
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose=1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))



if __name__ == '__main__':
    dataset = Dataset()
    dataset.load()
    model = Model()
#==============================================================================
#     model.load_model('./model/model0305.h5')
#     model.evaluate(dataset)
#==============================================================================
    model.build_model(dataset)
    model.train(dataset)
