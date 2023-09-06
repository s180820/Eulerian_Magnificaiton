#take in a video and calculate the ICA of each color channel of the video

import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
#from sklearn.decomposition import fastica

def get_colorchannels(frame):
    #get the color channels of the video
    blue_channel, green_channel, red_channel = cv2.split(frame)
    color_channels = [blue_channel, green_channel, red_channel]
    return color_channels

#def ICA(color_channel):
 #   #calculate the ICA of the color channel
  #  ica = fastica(color_channel)
   # return ica
    