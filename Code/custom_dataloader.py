import glob
from IPython.display import Video, HTML
import os
import cv2
import sys
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import json
from scipy import signal

sys.path.append('Code/Skin_segmentation')
sys.path.append('models')

import skin_detection_runfile
from rPPGNet import *

import helper_functions as helper_functions

class CustomDataset(Dataset):
    def __init__(self, root_dir, json_file, frames = 64):  # Adjust the resolution as needed
        self.root_dir = root_dir
        self.frames  = frames
        self.data = self.load_data(json_file)
    
    def load_data(self, json_file):
        with open(json_file) as file:
            data = json.load(file)
        return data
    
    def __len__(self):
        return len(self.data)

    def load_video_frames(self, video_path, bb_data):
        mask_array, frame_array = skin_detection_runfile.convert_video_with_progress(video_path, bb_data, frames = self.frames)
        return mask_array, frame_array
    
    def load_ecg_data(self, ecg_path, index_path):
        ecg = pd.read_csv(ecg_path)
        ecg[" ECG"] = signal.detrend(ecg[" ECG HR"])  # Detrending
        ecg["ECG_norm"] = (ecg[" ECG"] - ecg[" ECG"].mean()) / ecg[" ECG"].std()  # Normalizing
        ecg = ecg.iloc[::5, :]  # Choosing only the signal picked up by the vitacom
        ecg["ECG_MV"] = helper_functions.moving_average(ecg["ECG_norm"], 5)  # Calculate 5 point moving average
        
        idx = pd.read_csv(index_path, header=None, names=["timestamp", "idx_sig"])
        ecg = ecg.iloc[int(idx.iloc[0].idx_sig/5):]
        
        ecg = ecg["ECG_MV"][1:self.frames+1].values  # Get the first specified frames
        return ecg
    
    def __getitem__(self, idx):
        folder, sub_folder, video_file, ecg_file, index_file, bb_file = self.data[idx]
        video_path = os.path.join(self.root_dir, folder, sub_folder, video_file)
        ecg_path = os.path.join(self.root_dir, folder, sub_folder, ecg_file)
        index_path = os.path.join(self.root_dir, folder, sub_folder, index_file)
        bb_file = os.path.join(self.root_dir, "bbox", folder, sub_folder, "c920-1.face")
        bb_data = pd.read_csv(bb_file, sep=" ", header=None, names=["frame", "x", "y", "w", "h"]).drop("frame", axis=1)
        if "c920-1" not in video_path:
            video_path = os.path.join(self.root_dir, folder, sub_folder, data[folder][sub_folder]["video_2"])



        mask_array, frame_array,  = self.load_video_frames(video_path, bb_data)
        ecg_data = self.load_ecg_data(ecg_path, index_path)

        return video_tensor, ecg_tensor
# Define your data and DataLoader
json_file = 'Data/json_structure.json'
root_dir = "/work3/s174159/data/"
frames = 64
custom_dataset = CustomDataset(json_file, root_dir, frames=frames)
data_loader = DataLoader(custom_dataset, batch_size=1, shuffle=True)

# Iterate through the DataLoader to access your data
for i, (video_tensor, ecg_tensor) in enumerate(data_loader):
    print(f"Batch {i}, Video Tensor Shape: {video_tensor.shape}")
    print(f"Batch {i}, ECG Tensor Shape: {ecg_tensor.shape}")
    # Your training code here