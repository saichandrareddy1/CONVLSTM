#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 17:46:47 2019

@author: saireddy
"""

import cv2
import numpy as np
from keras.models import load_model


model = load_model("/home/saireddy/Action/LRCNN.h5")

model.summary()


classes = ['goal', 'FreeKick']
cap = cv2.VideoCapture("/home/saireddy/Action/Input.mp4")

FILE_OUTPUT = "/home/saireddy/Action/Trail3.avi"
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter(FILE_OUTPUT, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                      10, (frame_width, frame_height))


count = 0
#Q = deque(maxlen= "size")
while cap.isOpened():
    _, frame = cap.read()
    
    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    
    if frame is not None:
        output = frame.copy()
        frame_count = ("Frames:{}".format(count))
        count= count + 1
    else:
        print("sorry frames was ****completed****")
        break
    
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame, ((360//2)*2, 640//2), interpolation = cv2.INTER_CUBIC)
    frame1 = resized.reshape(1, 2, 320, 180, 3)

    
    pred_array = model.predict(frame1)
    print(pred_array)

    result = classes[np.argmax(pred_array)]
    
    
    score = float("%0.2f" % (max(pred_array[0]) * 100))

    
    text1 = ("Activity:{} |".format(result))
    cv2.putText(output, text1, (35, 50), cv2.FONT_HERSHEY_PLAIN, 1.25, (255, 255, 255), 2)
    cv2.rectangle(output, (20, 30), (590, 60), color=(0, 255, 0), thickness=2)
    
    text = ("Score:{} |".format(score))
    cv2.putText(output, text, (290, 50), cv2.FONT_HERSHEY_PLAIN, 1.25, (255, 255, 255), 2)
    cv2.putText(output, frame_count, (450, 50), cv2.FONT_HERSHEY_PLAIN, 1.25, (0, 255, 155), 2)
    
    print(f'Result: {result}, Score: {score}')
    
    cv2.imshow("frame", output)
    
    out.write(output)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()