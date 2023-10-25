# Import packages
import numpy as np
import os
import sys
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import json
from scipy import signal

# Add extra files that we need to use
sys.path.append('Code/Skin_segmentation')
sys.path.append('Models')

# Import classes
# import skin_detection_runefile as skin_driver
from skin_detection_runefile import MultipleVideoDriver
from rPPGNet import *
from helper_functions import helper_functions




class CustomDataset(Dataset):
    def __init__(self, root_dir, json_file, frames = 64):  # Adjust the number of frames depending on memory on GPUs
        self.root_dir = root_dir
        self.frames  = frames
        self.data = self.load_data(json_file)
        self._class = "[Custom dataset]"
    
    def load_data(self, json_file):
        with open(json_file) as file:
            data = json.load(file)
        return data
    
    def __len__(self):
        return len(self.data)

    def load_video_frames(self, video_path, bb_data):
        print(f"{self._class} Converting")
        mask_array, frame_array = MultipleVideoDriver.convert_video_with_progress(video_path, bb_data, frames = self.frames + 1)
        mask_array = helper_functions.binary_mask(mask_array)
        return mask_array, frame_array
    
    def load_ecg_data(self, ecg_path, index_path):
        idx = pd.read_csv(index_path, header = None, names = ["timestamp", "idx_sig"])
        ecg = pd.read_csv(ecg_path)
        ecg = helper_functions.smooth_ecg(ecg, idx)
        return ecg

    def getpaths(self, folder, sub_folder):
        video_path = os.path.join(self.root_dir, folder, sub_folder, self.data[folder][sub_folder]["video_1"])
        ecg_path = os.path.join(self.root_dir, folder, sub_folder, self.data[folder][sub_folder]["csv_1"])
        index_path = os.path.join(self.root_dir, folder, sub_folder, self.data[folder][sub_folder]["csv_2"])
        bb_file = os.path.join(self.root_dir, "bbox", folder, sub_folder, "c920-1.face")
        bb_data = pd.read_csv(bb_file, sep=" ", header=None, names=["frame", "x", "y", "w", "h"]).drop("frame", axis=1)

        if "c920-1" not in video_path:
            video_path = os.path.join(self.root_dir, folder, sub_folder, self.data[folder][sub_folder]["video_2"])

        return video_path, ecg_path, index_path, bb_file, bb_data

    
    def __getitem__(self, idx):
        folder = list(self.data.keys())[idx]  # Access the folder at the given index
        sub_folder = list(self.data[folder].keys())[0]  # Access the first sub-folder

        video_path, ecg_path, index_path, bb_file, bb_data = self.getpaths(folder, sub_folder)

        mask_array, frame_array,  = self.load_video_frames(video_path, bb_data)
        ecg = self.load_ecg_data(ecg_path, index_path)

        skin_seg_label, frame_tensor, ecg_tensor = helper_functions.tensor_transform(mask_array, frame_array, ecg, self.frames)
        
        assert frame_tensor.shape[1] == self.frames
        assert ecg_tensor.shape[0] == self.frames

        return skin_seg_label, frame_tensor, ecg_tensor
    
    def get_dataloader(self, batch_size = 1, *args, **kwargs):
        #if self.train:
         #   train_loader = DataLoader(dataset=self.train_subset, shuffle=True, batch_size=batch_size, *args, **kwargs)
          #  val_loader = DataLoader(dataset=self.val_subset, shuffle=False, batch_size=batch_size, *args, **kwargs)
           # return train_loader, val_loader
        #else:
        return DataLoader(self, batch_size=batch_size, shuffle=False, *args, **kwargs)
    
if __name__ == "__main__":
    # Define your data and DataLoader
    json_file = 'Data/json_structure'
    root_dir = "/work3/s174159/data/"
    frames = 64
    custom_dataset = CustomDataset(json_file=json_file, root_dir=root_dir, frames=frames)
    data_loader = DataLoader(custom_dataset, batch_size=1, shuffle=True)

    # Iterate through the DataLoader to access your data
    for i, (skin_seg_label, frame_tensor, ecg_tensor) in enumerate(data_loader):
        print(f"{custom_dataset._class} Batch {i}, skin_seg_label: {skin_seg_label}")
        print(f"{custom_dataset._class} Batch {i}, Video Tensor Shape: {frame_tensor.shape}")
        print(f"{custom_dataset._class} Batch {i}, ECG Tensor Shape: {ecg_tensor.shape}")
        # Your training code here