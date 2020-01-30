# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 16:48:40 2019

@author: SUPERMAN
"""

import cv2
import numpy as np
import os

#path =

fol= ['training', 'testing', 'validation']
classes = ['forehand_openstands','forehand_volley','forehead_slice','kick_service','slice_service', 'smash']


for path in fol:
	for clss in classes:
		paths = "/home/devil/Documents/orbitShifters/framed_data/"+path+"/"+clss+"/"""
		print('links',paths)
		li = os.listdir(paths)
		print(len(li))
		for image in li:
			img = cv2.imread(paths+image)
			if img is None:
				print(type(img))
				os.remove(paths+image)
				print(paths+image)
	   
