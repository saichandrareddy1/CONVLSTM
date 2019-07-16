from keras import applications
from keras.models import Model, Sequential
from keras.layers import Dense, Input, BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D, GlobalAveragePooling1D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Nadam, SGD, Adam
#from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers import Conv3D, Conv2D, MaxPool2D, Flatten, Dropout, Lambda
import os
import numpy as np
import keras.backend as K
from keras.preprocessing.image import img_to_array, load_img
import pandas as pd
from ImageGenerator_v2 import ImageDataGenerator


train_data_dir = "/home/saireddy/Action/TrainImages"
validation_data_dir = "/home/saireddy/Action/ValImages"
def obtain_datagen(datagen, train_path):
	return datagen.flow_from_directory(train_path, class_mode='binary',
                                    target_size=(320, 180), classes = ['goal', 'FreeKick'],
                                    batch_size=8, frames_per_step=2)

datagen = ImageDataGenerator(
		rescale=1./ 225,
		shear_range=0.2,
		zoom_range=0.2)

train_generator = obtain_datagen(datagen, train_data_dir)
validation_generator = obtain_datagen(datagen, validation_data_dir)

frames = 2
img_width = 320
img_height = 180
channels = 3


model = Sequential()
model.add(TimeDistributed(Conv2D(5, 2, 2,activation='relu'
                                 ,border_mode='valid'), 
          input_shape = (frames, img_width, img_height, channels)))
print(model.output_shape)
model.add(TimeDistributed(Flatten()))
print(model.output_shape)
#model.add(Dropout(0.5, input_shape = (16, 30)))
model.add(LSTM(3, return_sequences=False))
print(model.output_shape)
model.add(Dense(2, activation = 'sigmoid'))
model.summary()

optimizer = SGD(lr=0.01)
loss = 'sparse_categorical_crossentropy'	
model.compile(loss=loss,optimizer=optimizer, metrics=['accuracy'])

history = model.fit_generator(train_generator,
						steps_per_epoch=80,
						epochs=50,
						validation_data=validation_generator,
						validation_steps=40)

model.save("/home/saireddy/Action/LRCNN2.h5")

######Testing-----------------*****************
score = model.evaluate(X_train, y_train, steps=10)
print("Accuracyloss:-", score[0])
print("AccuracyScore",score[1] )

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