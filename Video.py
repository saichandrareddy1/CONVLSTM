#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:34:25 2019

@author: saireddy
"""

import numpy as np
import os
import cv2

def change_img_to_npy(path, new_path):
    for i, img in enumerate(os.listdir(path)):
        try:
            if img is not None:
                color = cv2.imread(path+img, cv2.IMREAD_COLOR)
                np.save(new_path+"{}.npy".format(str(i)),  color)
                print("INFO [], SUCESSS............")
        except:
            print("Except")

if __name__ == "__main__":
    path = "/home/saireddy/Action/TrainImages/goalresized/"
    new_path = "/home/saireddy/Action/TrainImages/newgoal/"
    change_img_to_npy(path, new_path)
    
def numpy_to_csv(path, path_train):
    Train_data = []
    for data_train in os.listdir(path_train):
        array_train = np.load(path_train+data_train,  allow_pickle = True)
        Train_data.append(array_train)
        
    arr = []
    for i in range(len(Train_data)):
        data = Train_data[i].reshape((50*50*3, -1))
        arr.append(data)
        
    with open(path, 'w') as fdata:
        try:
            for i in range(len(Train_data)):
                print(i)
                for j in range(arr[1].shape[0]):
                    if j != ((50*50*3) - 1):
                        fdata.write("{:.1f},".format(int(arr[i][j][0])))
                    elif j == ((50*50*3) - 1):
                        print("Hii", i)
                        fdata.write("{:.1f}\n".format(int(arr[i][j][0])))
        except:
            raise "ValueError:Sorry Proper Numpy array was not found"
            
if __name__ == "__main__":
    path = "/home/saireddy/Action/TrainImages/file.csv"
    path_train = "/home/saireddy/Action/TrainImages/newgoal/"
    numpy_to_csv(path, path_train)
    
    
    
import pandas as pd  
names = ["column{}".format(i) for i in range(0, ((50*50*3) - 1))]
data_set = pd.read_csv("/home/saireddy/Action/TrainImages/file.csv", header = None)
label = ["{}".format("goal") for i in range(0, data_set.shape[0])]
data_set['labels'] = label
data_set.to_csv("/home/saireddy/Action/TrainImages/Train.csv", index = False)

data = pd.read_csv("/home/saireddy/Action/ValImages/Test.csv", header = None)