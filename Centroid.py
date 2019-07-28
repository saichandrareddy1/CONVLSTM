import cv2
import time
import numpy as np
import math
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("/home/saireddy/Desktop/walk.mp4")
cascade = cv2.CascadeClassifier("/home/saireddy/Action/LSTM+CNN/haarcascade_frontalface_alt.xml")


#fps = cap.get(cv2.CAP_PROP_FPS)
#print ("Frames per second :{0}".format(fps))

FILE_OUTPUT = "/home/saireddy/Desktop/output2.mp4"
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter(FILE_OUTPUT, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                      10, (frame_width, frame_height))


spee = []
Distance = []
cex = []
cey = []
D = []
s = []
prev = []

def dist(centroid):
    for i in range(len(centroid)-1):
        print(centroid)
        dst = distance.euclidean(centroid[i], centroid[i+1])
        #dst = (dst)//0.03
        D.append(dst)
    return dst
    
#print("INFO[]... Please Enter **Width and Height in same Shape**")
#width = int(input("INFO[] Enter the width of the window:-"))
#height = int(input("INFO[] Enter the height of the window:-"))

centroid = []

while cap.isOpened():

    _, image = cap.read()
    
    if image is None:
        break

    #resize= cv2.resize(image, (width, height), interpolation = cv2.INTER_CUBIC)
    #image = resize.reshape(width, height, 3)

    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image = cv2.GaussianBlur(image, (5, 5), 0)
    #diff = cv2.absdiff(first_frame, image)

    #cv2.imshow("diff", diff)
    
    body = cascade.detectMultiScale(image, 1.3, 1)
    for (x, y, w, h) in body:
        rect = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if True:
            ceX = int(x+w/2)
            ceY = int(y+h/2)
            cex.append(ceX)
            cey.append(ceY)
            
            #print("Centoid", (ceX, ceY))

            print("length:-", len(centroid))
            
            if len(centroid) < 3:
                print("Appended", len(centroid))
                centroid.append((ceX, ceY))
            else:
                print("Length of the Centroid:-", len(centroid))
                continue
    
            if len(centroid) == 2:
                print("distance:", (dist(centroid)/30)*(3.6))
                cv2.putText(image, "distance:- {} km/hr".format(str((dist(centroid)/30)*3.6)), (x-25 , y-25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.circle(image, (int(x+w/2), int(y+h/2)), 5, (0, 255, 255), -1)
                cv2.putText(image, "centroid", (int(x+w/2) - 25, int(y+h/2) - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                print("Passed")
                continue
            
            if len(centroid) > 2:
                centroid.remove(centroid[0])
            else:
                print("Removed:-", centroid.remove(centroid[0]))
                continue
            
    cv2.imshow("image", image)
    out.write(image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
