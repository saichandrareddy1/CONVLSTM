import os
import cv2

class Frame_(object):
    def FrameCapture(self, path, video_arr):
        print(video_arr)
        count = 0
        for i in range(len(video_arr)):
            path_ = path + video_arr[i]
            print(path_)
            vidObj = cv2.VideoCapture(path_) 
            success = 1
            while success: 
                success, image = vidObj.read() 
                cv2.imwrite("/home/devil/Documents/orbitShifters/frames/frame%d.jpg" % count, image) 
                print(count)
                count += 1
    def frame_checker(self, path):
        for image in os.listdir(path):
            delete = False
            try:
                Image = cv2.imread(path+image)
                if Image is None:
                    delete = True
            except:
                print("Except")
                delete = True
        
            if delete:
                print("[INFO] deleting Image {}".format(path+image))
                os.remove(path+image)
        
if __name__ == '__main__': 
    filepath = "/home/devil/Documents/orbitShifters/sampleVideos/"
    path_ima = "/home/devil/Documents/orbitShifters/frames/"
    video_arr = []
    for video in os.listdir(filepath):
        video_arr.append(video)
    c = Frame_()
    c.FrameCapture(filepath, video_arr) 
    c.frame_checker(path_ima)
