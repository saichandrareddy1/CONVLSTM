#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[7]:


from tensorflow.keras import applications
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalAveragePooling1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.optimizers import Nadam, SGD, Adam, RMSprop
#from tensorflow.keras.preprocessing.image import ImageDataGenerator   #2D DATA
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import Conv3D, Conv2D, MaxPool2D, Flatten, Dropout, Lambda, ZeroPadding2D
import os
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import pandas as pd
from Generator import ImageDataGenerator ##3D DATA
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import matplotlib.pyplot as plt
#from Mail import MailSender


# In[3]:


datagen = ImageDataGenerator(
        rescale = 1./255,
        featurewise_center=True, 
        samplewise_center=True,
        rotation_range=40, width_shift_range=0.3,
        height_shift_range=0.2,
        shear_range=0.4, zoom_range=[0.2, 0.5],
        channel_shift_range=0.0, fill_mode='nearest', 
        cval=0.0, horizontal_flip=True, 
        vertical_flip=True,data_format=None)

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
### tennis function for model architecture
def tennis(n_model=Sequential()):
    
    model = n_model
    ###CONV LAYER
    model.add(TimeDistributed(Conv2D(3, 5, 1,activation='relu',padding='same'),
                              input_shape = (frames, img_width, img_height, channels)))
    print(model.output_shape)
    model.add(Dropout(dropout))
    
    ###CONV 2
    model.add(TimeDistributed(MaxPool2D((2,2))))
    model.add(BatchNormalization())
    model.add(TimeDistributed(ZeroPadding2D(1)))
    model.add(TimeDistributed(Conv2D(5, 5, 1,activation='relu'
                                     ,padding='same')))
    print(model.output_shape)

    ###CONV 3
    model.add(Dropout(dropout))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Conv2D(7, 5, 1,activation='relu'
                                     ,padding='same')))

    ##CONV 4
    model.add(TimeDistributed(MaxPool2D((2,2))))
    model.add(TimeDistributed(ZeroPadding2D(1)))
    model.add(TimeDistributed(Conv2D(13, 5, 1,activation='relu'
                                     ,padding='same')))

    ###CONV 5
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    print(model.output_shape)
    model.add(TimeDistributed(Conv2D(17, 5, 1,activation='relu'
                                     ,padding='same')))

    ####CONV 6
    model.add(TimeDistributed(MaxPool2D((2,2))))
    model.add(TimeDistributed(ZeroPadding2D(1)))
    model.add(TimeDistributed(Conv2D(20, 5, 1,activation='relu'
                                     ,padding='same')))

    ###CONV 7
    model.add(Dropout(dropout))
    model.add(TimeDistributed(MaxPool2D((2,2))))
    model.add(TimeDistributed(Conv2D(23, 3, 1,activation='relu'
                                     ,padding='same')))
    ###conv 8
    model.add(TimeDistributed(Conv2D(27, 3, 1,activation='relu'
                                     ,padding='same')))

    ####CONV 9
    model.add(Dropout(dropout))
    model.add(BatchNormalization())
    model.add(TimeDistributed(MaxPool2D((2,2))))
    model.add(TimeDistributed(Conv2D(32, 3, 1,activation='relu'
                                     ,padding='same'))) 
    ###conv 10
    model.add(TimeDistributed(Conv2D(35, 3, 1,activation='relu'
                                     ,padding='same')))


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
    return model.summary(), model

model, mainmodel=tennis(n_model=Sequential())
# Train the model for a specified number of epochs.
optimizer = SGD(lr=0.005, decay=1e-6/8, momentum=0.88, nesterov=True)


mainmodel.compile(loss= ['categorical_crossentropy'],
                  optimizer=optimizer,
                  metrics = ['accuracy'])
    
checkpoint = ModelCheckpoint("/home/saireddy/Action/Tennis/LRCNNTennis.h5", monitor='val_acc', verbose=1, 
                             save_best_only=True, save_weights_only=False, mode='auto', period=1)


history = mainmodel.fit_generator(train_generator, steps_per_epoch = 1,
                        epochs=1, validation_data=validation_generator,
                        validation_steps = 1,verbose=1,callbacks = [checkpoint])
        
# Evaluate the model with the eval dataset.
score  = model.evaluate_generator(test_generator, steps = 2)


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


# ##### Required Comments

# In[ ]:


### conv3 comment layers
#model.add(TimeDistributed(MaxPool2D((2,2))))
#model.add(TimeDistributed(ZeroPadding2D(1)))
### conv4 comment layers
#model.add(BatchNormalization())
### conv5 comment layers
#model.add(TimeDistributed(ZeroPadding2D(1)))
### con6 comment layers
#model.add(BatchNormalization())
### con7 comment layers
#model.add(BatchNormalization())
#optimizer = RMSprop(lr=0.005, rho=0.8, epsilon=1e-10, decay=1e-10)
#optimizer = Adam(lr = 0.001, beta_1=0.8, beta_2=0.999,epsilon=None,decay=0.0, amsgrad=True)
#early = EarlyStopping(monitor='val_acc', min_delta=0, patience=15, verbose=1, mode='auto')

