#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 02:36:51 2019

@author: saireddy
"""

__author__ = "Sai Reddy"
from keras import applications
from keras.models import Model, Sequential
from keras.layers import Dense, Input, BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D, GlobalAveragePooling1D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Nadam, SGD, Adam, RMSprop
#from keras.preprocessing.image import ImageDataGenerator   #2D DATA
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers import Conv3D, Conv2D, MaxPool2D, Flatten, Dropout, Lambda, ZeroPadding2D
import os
import numpy as np
import keras.backend as K
from keras.preprocessing.image import img_to_array, load_img
import pandas as pd
from Generator import ImageDataGenerator ##3D DATA
from keras import regularizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from Mail import MailSender


datagen = ImageDataGenerator(
        rescale = 1./255,
        featurewise_center=True, 
        samplewise_center=True,
        rotation_range=40, width_shift_range=0.3,
        height_shift_range=0.2,
        shear_range=0.4, zoom_range=[0.2, 0.5],
        channel_shift_range=0.0, fill_mode='nearest', 
        cval=0.0, horizontal_flip=True, vertical_flip=True,
        data_format=None)

frames = 6

train_generator = datagen.flow_from_directory("/home/saireddy/Action/Tennis/TrainImages", class_mode='categorical',
                                        target_size=(320, 180), 
                                        classes = ['backhand', 'smash'],
                                        batch_size=8, frames_per_step=frames)
    
validation_generator = datagen.flow_from_directory("/home/saireddy/Action/Tennis/ValImages", class_mode='categorical',
                                        target_size=(320, 180),
                                        classes = ['backhand', 'smash'],
                                        batch_size=8, frames_per_step=frames)

test_generator = datagen.flow_from_directory("/home/saireddy/Action/Tennis/TestImages", class_mode='categorical',
                                        target_size=(320, 180),
                                        classes = ['backhand', 'smash'],
                                        batch_size=8, frames_per_step=frames)
    

img_width = 320
img_height = 180
channels = 3
dropout = 0.5
batch_size = 8
print(batch_size, frames, img_width, img_height, channels)


#MODEL .........-----------**************************##
model = Sequential()
    
###CONV LAYER
model.add(TimeDistributed(Conv2D(3, 5, 1,activation='relu'
                                     ,border_mode='same'), 
              input_shape = (frames, img_width, img_height, channels)))
print(model.output_shape)
model.add(Dropout(dropout))
    
###CONV2
model.add(TimeDistributed(MaxPool2D((2,2))))
model.add(BatchNormalization())
model.add(TimeDistributed(ZeroPadding2D(1)))
model.add(TimeDistributed(Conv2D(5, 5, 1,activation='relu'
                                     ,border_mode='same')))
print(model.output_shape)

###CONV3
model.add(Dropout(dropout))
#model.add(TimeDistributed(MaxPool2D((2,2))))
model.add(BatchNormalization())
#model.add(TimeDistributed(ZeroPadding2D(1)))
model.add(TimeDistributed(Conv2D(7, 5, 1,activation='relu'
                                     ,border_mode='same')))

##CONV 4
model.add(TimeDistributed(MaxPool2D((2,2))))
#model.add(BatchNormalization())
model.add(TimeDistributed(ZeroPadding2D(1)))
model.add(TimeDistributed(Conv2D(13, 5, 1,activation='relu'
                                     ,border_mode='same')))

###CONV 5
model.add(BatchNormalization())
model.add(Dropout(dropout))
print(model.output_shape)
#model.add(TimeDistributed(ZeroPadding2D(1)))
model.add(TimeDistributed(Conv2D(17, 5, 1,activation='relu'
                                     ,border_mode='same')))

####CONV 6
model.add(TimeDistributed(MaxPool2D((2,2))))
#model.add(BatchNormalization())
model.add(TimeDistributed(ZeroPadding2D(1)))
model.add(TimeDistributed(Conv2D(20, 5, 1,activation='relu'
                                     ,border_mode='same')))

###CONV 7
model.add(Dropout(dropout))
#model.add(BatchNormalization())
model.add(TimeDistributed(MaxPool2D((2,2))))
model.add(TimeDistributed(Conv2D(23, 3, 1,activation='relu'
                                     ,border_mode='same')))
###conv 8
model.add(TimeDistributed(Conv2D(27, 3, 1,activation='relu'
                                     ,border_mode='same')))

####CONV 9
model.add(Dropout(dropout))
model.add(BatchNormalization())
model.add(TimeDistributed(MaxPool2D((2,2))))
model.add(TimeDistributed(Conv2D(32, 3, 1,activation='relu'
                                     ,border_mode='same')))
###conv 10
model.add(TimeDistributed(Conv2D(35, 3, 1,activation='relu'
                                     ,border_mode='same')))


####FLATTEN LAYER
model.add(TimeDistributed(Flatten()))
print(model.output_shape)
model.add(Dropout(dropout))

####LSTM LAYER
model.add(LSTM(10, return_sequences=False))
    
###DENSE LAYER1
model.add(Dropout(dropout))
print(model.output_shape)
model.add(BatchNormalization())
model.add(Dropout(dropout))
model.add(Dense(500, activation='relu'))

####DENSE LAYER2
model.add(BatchNormalization())
model.add(Dropout(dropout))
model.add(Dense(250, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(dropout))

#####DENSE LAYER3
model.add(Dense(150, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(dropout))

####OUTPUT LAYER
model.add(Dense(2 , activation = 'softmax'))
model.summary()


# Train the model for a specified number of epochs.
optimizer = SGD(lr=0.005, decay=1e-6/8, momentum=0.88, nesterov=True)
#optimizer = RMSprop(lr=0.005, rho=0.8, epsilon=1e-10, decay=1e-10)
#optimizer = Adam(lr = 0.001, beta_1=0.8, beta_2=0.999,epsilon=None,decay=0.0, amsgrad=True)

model.compile(loss= ['categorical_crossentropy'],
                  optimizer=optimizer,
                  metrics = ['accuracy'])
    
checkpoint = ModelCheckpoint("/home/saireddy/Action/Tennis/LRCNNTennis.h5", monitor='val_acc', verbose=1, 
                             save_best_only=True, save_weights_only=False, mode='auto', period=1)
#early = EarlyStopping(monitor='val_acc', min_delta=0, patience=15, verbose=1, mode='auto')
epochs = 10
num = int(input("INFO[] ENTER NUMBER WHICH YOU NEED TO DIVIDE:-"))
for epoch in range(epochs):
	history = model.fit_generator(train_generator, steps_per_epoch = 1,
	                    epochs=1, validation_data=validation_generator,
	                    validation_steps = 1, verbose=1,
	                    callbacks = [checkpoint])
	
	   
	if (epoch)%num == 0:
		val_acc = history.history['val_acc']
		val_acc = val_acc[-1]
		acc = history.history['acc']
		acc = acc[-1]
		mail = MailSender("Message", "Sender_mail@gmail.com", "password", "Reciver_mail@gmail.com", 47, val_acc, acc, epoch, num) 
		mail.mailsend()
		
# Evaluate the model with the eval dataset.
score  = model.evaluate_generator(test_generator, steps = 2)

import matplotlib.pyplot as plt
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
    
   
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

    
   
print('Test loss:', score[0])
print('Test accuracy:', score[1])
