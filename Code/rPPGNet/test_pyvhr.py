import pyVHR as vhr
import numpy as np
import cv2
from pyVHR.utils.errors import getErrors, printErrors, displayErrors, BVP_windowing
import os

root_dir = "/work3/s174159/data/"
video_file = root_dir + "00/02/c920-1.avi"
bb_file = root_dir + "bbox/00/02/c920-1.face"
ecg_path = root_dir + "00/02/viatom-raw.csv"
index_path = root_dir + "00/02/c920.csv"

# params
wsize = 10                  # window size in seconds
roi_approach = 'patches'   # 'holistic' or 'patches'
bpm_est = 'clustering'     # BPM final estimate, if patches choose 'medians' or 'clustering'
method = 'cpu_OMIT'       # one of the methods implemented in pyVHR

# run
# pipe = Pipeline()          # object to execute the pipeline
# bvps, timesES, bpmES = pipe.run_on_video(video_file,
#                                         winsize=wsize, 
#                                         roi_method='convexhull',
#                                         roi_approach=roi_approach,
#                                         method=method,
#                                         estimate=bpm_est,
#                                         patch_size=40, 
#                                         RGB_LOW_HIGH_TH=(5,230),
#                                         Skin_LOW_HIGH_TH=(5,230),
#                                         pre_filt=True,
#                                         post_filt=True,
#                                         cuda=True, 
#                                         verb=True)

fps = 30


sp = vhr.extraction.sig_processing.SignalProcessing()
frames = sp.extract_raw(video_file)

bvp_pred = vhr.deepRPPG.HR_CNN_bvp_pred(frames)
bvps = vhr.BPM.BVPsignal(bvp_pred, fps)

bvp_win, timesES = BVP_windowing(bvp_pred, wsize, fps, stride=1)
bpmES = vhr.BPM.BVP_to_BPM(bvp_win, fps) 

print(bpmES)