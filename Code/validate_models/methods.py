import numpy as np
import torch
import torchvision.transforms as transforms
import time
from collections import OrderedDict
from torch.utils.data import DataLoader
from utils import butter_bandpass_filter, detrend, preprocess_raw_video
from PulseDataset import PulseDataset
from FaceHRNet09V4ELU import FaceHRNet09V4ELU
from MTTS_CAN import MTTS_CAN
import os
import requests
import scipy
from scipy.signal import butter
import tensorflow as tf


def HR_CNN_bvp_pred(frames):
    print("initialize model...")

    model_path = 'models/hr_cnn_model.pth'
    if not os.path.isfile(model_path):
      print('Downloading HR_CNN model...')
      url = "https://github.com/phuselab/pyVHR/raw/master/resources/deepRPPG/hr_cnn_model.pth"
      r = requests.get(url, allow_redirects=True)
      if not os.path.isdir('models'):
        os.mkdir('models')
      open(model_path, 'wb').write(r.content)   
      print("Downloaded HR_CNN model to: ", model_path)

    model = FaceHRNet09V4ELU(rgb=True)

    model = torch.nn.DataParallel(model)

    #model.cuda()

    ss = sum(p.numel() for p in model.parameters())
    print('num params: ', ss)

    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    new_state_dict = OrderedDict()
    # original saved file with DataParallel
    for k, v in state_dict.items():
        new_state_dict['module.' + k] = v

    model.load_state_dict(new_state_dict)

    pulse_test = PulseDataset(frames, transform=transforms.ToTensor())

    val_loader = DataLoader(
        pulse_test,
        batch_size=128, shuffle=False, pin_memory=True, drop_last=True)

    model.eval()

    outputs = []

    start = time.time()
    #computing the output
    for i, net_input in enumerate(val_loader):
        print("processing batch: ", i, "/", len(val_loader), "...")
        with torch.no_grad():
            output = model(net_input)
            outputs.append(output.squeeze())

    end = time.time()
    print("processing time: ", end - start)

    outputs = torch.cat(outputs)

    outputs = (outputs - torch.mean(outputs)) / torch.std(outputs)

    outputs = outputs.tolist()

    fs = 30
    lowcut = 0.8
    highcut = 6

    filtered_outputs = butter_bandpass_filter(outputs, lowcut, highcut, fs, order=4)
    filtered_outputs = (filtered_outputs - np.mean(filtered_outputs)) / np.std(filtered_outputs)

    return np.array(filtered_outputs)

def MTTS_CAN_deep(frames, fs, model_checkpoint=None, batch_size=100, dim=36, img_rows=36, img_cols=36, frame_depth=10, verb=0, filter_pred=False):

  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
     for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,True)

  if model_checkpoint is None:
    model_checkpoint = 'models/mtts_can_model.hdf5'
    if not os.path.isfile(model_checkpoint):
      url = "https://github.com/phuselab/pyVHR/raw/master/resources/deepRPPG/mtts_can_model.hdf5"
      print('Downloading MTTS_CAN model...')
      r = requests.get(url, allow_redirects=True)
      if not os.path.isdir('models'):
        os.mkdir('models')
      open(model_checkpoint, 'wb').write(r.content)   

  # frame preprocessing
  dXsub = preprocess_raw_video(frames, fs=fs, dim=dim)
  dXsub_len = (dXsub.shape[0] // frame_depth)  * frame_depth
  dXsub = dXsub[:dXsub_len, :, :, :]

  # load pretrained model
  model = MTTS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
  model.load_weights(model_checkpoint)

  # apply pretrained model
  yptest = model.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=verb)

  # filtering
  
  pulse_pred = yptest[0]
  pulse_pred = detrend(np.cumsum(pulse_pred), 100)
  if filter_pred:
    [b_pulse, a_pulse] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    #[b_pulse, a_pulse] = butter(1, [0.65 / fs * 2, 2.5 / fs * 4], btype='bandpass')
    pulse_pred = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(pulse_pred))
  return pulse_pred
