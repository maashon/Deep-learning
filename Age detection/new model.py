# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 08:46:17 2018

@author: Parsa
"""

import glob
import cv2
import numpy as np
from keras.utils import np_utils

length=100;

data = []
print("loading training data images:\n")
for i in range(1,32):
    
    path = "C:/Users/Parsa/Desktop/deeep learning/dataset/mix with age/{}/".format(i)
    images_paths = glob.glob(path +"*.png") + glob.glob(path +"*.jpg")
    images_paths.sort()

    info_path = glob.glob(path +"*.txt")
    info_file = open(info_path[0],'r');
    
    age =    int(float(info_file.readline().split(":")[1].replace("\n","").strip()))
   
    
   
    for img in images_paths:
        image = cv2.imread(img)
        image = cv2.resize(image,(length,length));
        image = image / np.max(image)
        image = image.astype(np.float32)
        
        imageEntity = {"image" : image,"age" : age}
        data.append(imageEntity)
        
x_train_image =[]
y_train_age = []

for i in data:
    x_train_image.append(i['image'])
    y_train_age.append(i['age'])
    
y_train_age = np_utils.to_categorical(y_train_age)   
x_train_image =np.array(x_train_image)
print("data loaded successfully!")
print(x_train_image.shape)

#   ------------------- loading test data
test_data = []
print("loading test data images:\n")
for i in range(36,48):
    
    path = "C:/Users/Parsa/Desktop/deeep learning/dataset/test imgs/{}/".format(i)
    images_paths = glob.glob(path +"*.png") + glob.glob(path +"*.jpg")
    images_paths.sort()

    info_path = glob.glob(path +"*.txt")
    info_file = open(info_path[0],'r');
    
    age =    int(float(info_file.readline().split(":")[1].replace("\n","").strip()))
   
    
   
    for img in images_paths:
        image = cv2.imread(img)
        image = cv2.resize(image,(length,length));
        image = image / np.max(image)
        image = image.astype(np.float32)
        
        imageEntity = {"image" : image,"age" : age}
        test_data.append(imageEntity)

x_test_image =[]
y_test_age = []

for i in test_data:
    x_test_image.append(i['image'])
    y_test_age.append(i['age'])
 
#x_train_image.reshape(20,40,40,3);
y_test_age = np_utils.to_categorical(y_test_age)   
x_test_image =np.array(x_test_image)
print("test data loaded successfully! ")
print(x_test_image.shape)
x_train_image=x_train_image[:,:,:,0]
x_test_image=x_test_image[:,:,:,0]
x_train = x_train_image.reshape(124,length*length)
x_test = x_test_image.reshape(48,length*length)
y_train=y_train_age
y_test=y_test_age

#making the model

from keras.models import Sequential
from keras.layers import Dense
from keras.activations import relu
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
print("making the fully connected model:")
my_model=Sequential()
my_model.add(Dense(500,activation='relu',input_shape=(length*length,)))
my_model.add(Dense(100,activation='relu'))
my_model.add(Dense(60,activation=relu))
my_model.summary()
my_model.compile(optimizer=SGD(lr=0.001),loss=categorical_crossentropy,metrics=['accuracy'])
print("making fully connected model completed")


"""training section"""

train_history=my_model.fit(x_train,y_train,epochs=100)