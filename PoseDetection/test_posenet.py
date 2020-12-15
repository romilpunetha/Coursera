import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import cv2 as cv
import math
import numpy as np
from os import listdir
from os.path import isfile, join


def FrameCapture(path): 
      
    frames = [] 

    # Path to video file 
    vidObj = cv.VideoCapture(path) 
  
    # Used as counter variable 
    count = 0
  
    # checks whether frames were extracted 
    success = 1
  
    while success: 
  
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 
  
        # Saves the frames with frame-count 
        try: 
            img2 = cv.resize(image, (257, 257))
            frames.append(img2)
        except Exception as e:
            print(str(e))
        # cv.imwrite("frame%d.jpg" % count, image) 
        count += 1

    return frames


model_path = "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]


input_path = "data/"
onlyfiles = [f for f in listdir(input_path) if isfile(join(input_path, f))]

def parse _output(heatmap_data,offset_data, threshold):

  '''
  Input:
    heatmap_data - hetmaps for an image. Three dimension array
    offset_data - offset vectors for an image. Three dimension array
    threshold - probability threshold for the keypoints. Scalar value
  Output:
    array with coordinates of the keypoints and flags for those that have
    low probability
  '''

  joint_num = heatmap_data.shape[-1]
  pose_kps = np.zeros((joint_num,3), np.uint32)

  for i in range(heatmap_data.shape[-1]):

      joint_heatmap = heatmap_data[...,i]
      max_val_pos = np.squeeze(np.argwhere(joint_heatmap==np.max(joint_heatmap)))
      remap_pos = np.array(max_val_pos/8*257,dtype=np.int32)
      pose_kps[i,0] = int(remap_pos[0] + offset_data[max_val_pos[0],max_val_pos[1],i])
      pose_kps[i,1] = int(remap_pos[1] + offset_data[max_val_pos[0],max_val_pos[1],i+joint_num])
      max_prob = np.max(joint_heatmap)

      if max_prob > threshold:
        if pose_kps[i,0] < 257 and pose_kps[i,1] < 257:
          pose_kps[i,2] = 1

  return pose_kps

def draw_kps(show_img,kps, ratio=None):
    for i in range(5,kps.shape[0]):
      if kps[i,2]:
        if isinstance(ratio, tuple):
          cv.circle(show_img,(int(round(kps[i,1]*ratio[1])),int(round(kps[i,0]*ratio[0]))),2,(0,255,255),round(int(1*ratio[1])))
          continue
        cv.circle(show_img,(kps[i,1],kps[i,0]),2,(0,255,255),-1)
    return show_img

for f in onlyfiles:
    input_file = f
    input_video = input_path + input_file
    frames = FrameCapture(input_video)
    height,width,layers=frames[1].shape
    fourcc = cv.VideoWriter_fourcc(*'mp4v') 
    try:
        video=cv.VideoWriter('output/' + input_file,fourcc,20,(width,height))
        for frame in frames:
            try:
                template_image_src = frame
                template_image = cv.resize(template_image_src, (width, height))
                template_input = np.expand_dims(template_image.copy(), axis=0)
                floating_model = input_details[0]['dtype'] == np.float32
                if floating_model:
                    template_input = (np.float32(template_input) - 127.5) / 127.5
                interpreter.set_tensor(input_details[0]['index'], template_input)
                interpreter.invoke()
                template_output_data = interpreter.get_tensor(output_details[0]['index'])
                template_offset_data = interpreter.get_tensor(output_details[1]['index'])
                # Getting rid of the extra dimension
                template_heatmaps = np.squeeze(template_output_data)
                template_offsets = np.squeeze(template_offset_data)
                template_show = np.squeeze((template_input.copy()*127.5+127.5)/255.0)
                template_show = np.array(template_show*255,np.uint8)
                template_kps = parse_output(template_heatmaps,template_offsets,0.3)

                # pose_landmarks, annotated_image = pose_tracker.run(input_frame=frame)
                video.write(draw_kps(template_show.copy(),template_kps))
            except Exception as e2:
                print(e2)
    except Exception as e1:
        print(e1)
    finally:
        cv.destroyAllWindows()
        video.release()
