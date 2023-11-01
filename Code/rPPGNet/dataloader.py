# Import packages
import numpy as np
import os
import sys
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import json
from scipy import signal


pd.options.mode.chained_assignment = None

# Add extra files that we need to use
sys.path.append('Code/Skin_segmentation')
sys.path.append('Models')

# Import classes
# import skin_detection_runefile as skin_driver
from skin_video_driver import MultipleVideoDriver
from rPPGNet import *
from helper_functions import helper_functions
import random


class CustomDataset(Dataset):
    def __init__(self, root_dir, frames = 64, verbosity = False, purpose="train"):  # Adjust the number of frames depending on memory on GPUs
        self.root_dir = root_dir
        self.frames = frames
        self.video_paths, self.ecg_paths, self.idx_paths, self.bb_paths = self.load_data()
        self.videoDriver = MultipleVideoDriver()
        #self.data = pd.DataFrame(data={"videos": self.video_paths, 
         #                                   "ecg" : self.ecg_paths, 
          #                                  "idx" :self.idx_paths, 
           #                                 "bb": self.bb_paths})
        self._class = "[Custom dataset]"
        self.start_frame_idx = 1  # Change this to start from the first frame
        self.current_frame_idx = self.start_frame_idx
        self.videoECG_counter = 0
        self.verbosity = verbosity
        self.purpose = purpose
        self.random_list = random.sample(range(82), 82)
        self.train_subset, self.test_subset, self.val_subset = torch.utils.data.random_split(
        self, [0.7, 0.2, 0.1], generator=torch.Generator().manual_seed(1))
    
    def load_data(self):
        video_paths, ecg_paths, idx_paths, bb_paths = helper_functions.generate_list_of_video_paths()
        return video_paths, ecg_paths, idx_paths, bb_paths
    
    def __len__(self):
        return (helper_functions.get_total_frame_count(self.video_paths) // self.frames)  # Return the total number of videos

    def load_video_frames(self, video_path, bb_data, cur_frame_idx):
        print(f"{self._class} Converting")

        #output_dir = '/zhome/01/d/127159/Desktop/Eulerian_Magnificaiton/output_dir/Skin_segmentation/'
        #output_file = os.path.join(output_dir, str(self.videoECG_counter) +".mp4")
        mask_array, frame_array = self.videoDriver.convert_video_with_progress(video_file = video_path, data = bb_data,
                                                                              frames_to_process=self.frames + 1,
                                                                              starting_frame=cur_frame_idx, verbosity=False)
        mask_array = helper_functions.binary_mask(mask_array)
        return mask_array, frame_array
    
    def load_ecg_data(self, ecg_path, index_path, start_frame, end_frame):
        idx = pd.read_csv(index_path, header=None, names=["timestamp", "idx_sig"])
        ecg = pd.read_csv(ecg_path)
        ecg = helper_functions.smooth_ecg(ecg, idx)
        ecg = ecg[start_frame:end_frame+1]  # Fit ECG to frames.
        return ecg
    
    def getpaths(self):
        video_path = self.video_paths[self.random_list[self.videoECG_counter]]
        ecg_path = self.ecg_paths[self.random_list[self.videoECG_counter]]
        idx_path = self.idx_paths[self.random_list[self.videoECG_counter]]
        bb_path = self.bb_paths[self.random_list[self.videoECG_counter]]
        bb_data = pd.read_csv(bb_path, sep=" ", header=None, names=["frame", "x", "y", "w", "h"]).drop("frame", axis=1)
        return video_path, ecg_path, idx_path, bb_path, bb_data

    def __getitem__(self, idx):
        video_path, ecg_path, index_path, _ , bb_data = self.getpaths()
        # Calculate start and end frame indices for the current video
        start_frame = self.current_frame_idx
        video_frame_count = helper_functions.get_video_frame_count(video_path)
        end_frame = start_frame + min(self.frames, video_frame_count-start_frame)
        #end_frame = self.current_frame_idx + self.frames
        if self.verbosity:
            print(f"{self._class} [INFO]: VideoFile: {video_path} | ECGFile: {ecg_path} | IndexFile: {index_path}")
            print(f"{self._class} [INFO]: VideoCounter: {self.videoECG_counter} | FrameCounter: {start_frame} | TotalFrames: {video_frame_count}")


        # Load video frames and ECG data
        mask_array, frame_array = self.load_video_frames(video_path, bb_data, cur_frame_idx=start_frame)
        ecg = self.load_ecg_data(ecg_path, index_path, start_frame, end_frame)
        

        # Transform the data into tensors as needed
        skin_seg_label, frame_tensor, ecg_tensor = helper_functions.tensor_transform(mask_array, frame_array, ecg, self.frames)
        #print(ecg_tensor)

        # Update the current frame index
        self.current_frame_idx = end_frame
        if (not frame_tensor.shape[1] == self.frames) or (not ecg_tensor.shape[0] == self.frames):
            #print("Length of ecg GT", ecg_tensor.shape[0])
            #print("Number of frames:", frame_tensor.shape[1])
            if self.verbosity:
                print(f"{self._class} [DEBUGGING]", start_frame)
                print(f"{self._class} [DEBUGGING] Video frames available", video_frame_count)
                #for i in range(5000):
                print(f"{self._class} [INFO] GETTING NEW VIDEO!")
            self.videoECG_counter += 1
            self.current_frame_idx = self.start_frame_idx
            #return torch.FloatTensor(), torch.FloatTensor(), torch.FloatTensor()
            #self.__getitem__(idx)
        
        #assert frame_tensor.shape[1]> self.frames # Is allowed to be smaller - due to end of video.
        #assert ecg_tensor.shape[0]>  self.frames # Is allowed to be smaller - due to end of video.

        # Check if there are more frames in the current video for the next iteration.
        if (end_frame + self.frames >= video_frame_count):
            # Move to the next video
            self.videoECG_counter += 1
            self.current_frame_idx = self.start_frame_idx 

        return skin_seg_label, frame_tensor, ecg_tensor
        
    
    def get_dataloader(self, batch_size = 1, *args, **kwargs):
        if self.purpose == "train":
            return DataLoader(dataset=self.train_subset, shuffle=True, batch_size=batch_size, *args, **kwargs)
        if self.purpose == "test":
            return DataLoader(dataset=self.test_subset, shuffle=False, batch_size=batch_size, *args, **kwargs)
        if self.purpose == "val":
            return DataLoader(dataset=self.val_subset, shuffle=False, batch_size=batch_size, *args, **kwargs)
        #else:
        #return DataLoader(self, batch_size=batch_size, shuffle=False, *args, **kwargs)


class CustomDataset_OLD(Dataset):
    def __init__(self, root_dir, json_file, frames = 64, train=True, purpose="train"):  # Adjust the resolution as needed
        self.root_dir = root_dir
        self.frames  = frames
        self.data = self.load_data(json_file)
        self.train = train
        self.videoDriver = MultipleVideoDriver()

        if self.train:
            self.train_subset, self.val_subset = torch.utils.data.random_split(
        self, [0.8, 0.2], generator=torch.Generator().manual_seed(1))
    
    def load_data(self, json_file):
        with open(json_file) as file:
            data = json.load(file)
        return data
    
    def __len__(self):
        return len(self.data)

    def load_video_frames(self, video_path, bb_data):
        print("Converting")
        mask_array, frame_array = self.videoDriver.convert_video_with_progress(video_file = video_path, data = bb_data,
                                                                              frames_to_process=self.frames + 1,
                                                                              starting_frame=1, verbosity=False)
        mask_array = helper_functions.binary_mask(mask_array)
        return mask_array, frame_array
    
    def load_ecg_data(self, ecg_path, index_path):
        idx = pd.read_csv(index_path, header = None, names = ["timestamp", "idx_sig"])
        ecg = pd.read_csv(ecg_path)
        ecg = helper_functions.smooth_ecg(ecg, idx)
        return ecg
    
    def __getitem__(self, idx):
        folder = list(self.data.keys())[idx]  # Access the folder at the given index
        sub_folder = list(self.data[folder].keys())[0]  # Access the first sub-folder

        video_path = os.path.join(self.root_dir, folder, sub_folder, self.data[folder][sub_folder]["video_1"])
        ecg_path = os.path.join(self.root_dir, folder, sub_folder, self.data[folder][sub_folder]["csv_1"])
        index_path = os.path.join(self.root_dir, folder, sub_folder, self.data[folder][sub_folder]["csv_2"])
        bb_file = os.path.join(self.root_dir, "bbox", folder, sub_folder, "c920-1.face")
        bb_data = pd.read_csv(bb_file, sep=" ", header=None, names=["frame", "x", "y", "w", "h"]).drop("frame", axis=1)

        if "c920-1" not in video_path:
            video_path = os.path.join(self.root_dir, folder, sub_folder, self.data[folder][sub_folder]["video_2"])

        mask_array, frame_array,  = self.load_video_frames(video_path, bb_data)
        ecg = self.load_ecg_data(ecg_path, index_path)

        skin_seg_label, frame_tensor, ecg_tensor = helper_functions.tensor_transform(mask_array, frame_array, ecg, self.frames)
        
        assert frame_tensor.shape[1] == self.frames
        assert ecg_tensor.shape[0] == self.frames

        return skin_seg_label, frame_tensor, ecg_tensor
    
    def get_dataloader(self, batch_size = 1, *args, **kwargs):
        if self.train:
            train_loader = DataLoader(dataset=self.train_subset, shuffle=True, batch_size=batch_size, *args, **kwargs)
            val_loader = DataLoader(dataset=self.val_subset, shuffle=False, batch_size=batch_size, *args, **kwargs)
            return train_loader, val_loader
        else:
            return DataLoader(self, batch_size=batch_size, shuffle=False, *args, **kwargs)
    
if __name__ == "__main__":
    # Define your data and DataLoader
    json_file = 'Data/json_structure'
    root_dir = "/work3/s174159/data/"
    frames = 200
    custom_dataset = CustomDataset(root_dir=root_dir, frames=frames, verbosity=True)
    data_loader = DataLoader(custom_dataset, batch_size=1, shuffle=True)
    l_loader = len(data_loader)
    # Iterate through the DataLoader to access your data
    for i, (skin_seg_label, frame_tensor, ecg_tensor) in enumerate(data_loader):
        # print(f"{custom_dataset._class} Batch {i}, skin_seg_label: {skin_seg_label}")
        # print(f"{custom_dataset._class} Batch {i}, Video Tensor Shape: {frame_tensor.shape}")
        # print(f"{custom_dataset._class} Batch {i}, ECG Tensor Shape: {ecg_tensor.shape}")
       # print(f"{custom_dataset._class} Batch {i}")
       print(f"Loader itertation: {i}/{l_loader}")
        #break
        # Your training code here