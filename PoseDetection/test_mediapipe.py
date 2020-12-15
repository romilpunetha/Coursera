import mediapipe as mp
import numpy as np
import cv2 


def FrameCapture(path): 
      
    frames = [] 

    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
  
    # Used as counter variable 
    count = 0
  
    # checks whether frames were extracted 
    success = 1
  
    while success: 
  
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 
  
        # Saves the frames with frame-count 
        frames.append(image)
        # cv2.imwrite("frame%d.jpg" % count, image) 
        count += 1

    return frames

from os import listdir
from os.path import isfile, join

pose_tracker = mp.examples.UpperBodyPoseTracker()
input_path = "data/"
onlyfiles = [f for f in listdir(input_path) if isfile(join(input_path, f))]

for f in onlyfiles:
    input_file = f
    input_video = input_path + input_file
    frames = FrameCapture(input_video)
    height,width,layers=frames[1].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    try:
        video=cv2.VideoWriter('output/' + input_file,fourcc,20,(width,height))
        for frame in frames:
            try:
                pose_landmarks, annotated_image = pose_tracker.run(input_frame=frame)
                video.write(annotated_image)
            except Exception as e2:
                print(e2)
    except Exception as e1:
        print(e1)
    finally:
        cv2.destroyAllWindows()
        video.release()
