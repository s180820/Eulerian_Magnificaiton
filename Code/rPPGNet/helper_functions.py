from scipy import signal
import pandas as pdf
import numpy as np
import torch
import cv2
import json
import os


class helper_functions: 
      def moving_average(series, window_size=5):
            # Convert array of integers to pandas series 

            # Get the window of series 
            # of observations of specified window size 
            windows = series.rolling(window_size) 

            # Create a series of moving 
            # averages of each window 
            moving_averages = windows.mean() 

            # Convert pandas series back to list 
            moving_averages_list = moving_averages.tolist() 
            final_list = moving_averages_list

            return final_list
      def check_files_exist(file_path1, file_path2, file_path3, file_path4):
            if os.path.exists(file_path1) and os.path.exists(file_path2) and os.path.exists(file_path3) and os.path.exists(file_path4):
                  return True
            else:
                  return False
            
      def ecg_skip_signal(ecg, start_frame, end_frame):
            if len(ecg) < start_frame+end_frame:
                  return False

            
      def generate_list_of_video_paths(json_path = "Data/json_structure", root_dir = "/work3/s174159/data/"):
            with open(json_path) as json_file:
                  data = json.load(json_file)
            video_paths = []
            ecg_paths = [] 
            idx_paths = []
            bb_data_paths = []
            for folder in data:
                  for sub_folder in data[folder]:
                        video_path = root_dir + "{}/{}/{}".format(folder, sub_folder, data[folder][sub_folder]["video_1"])
                        ecg_path = root_dir + "{}/{}/{}".format(folder, sub_folder, data[folder][sub_folder]["csv_1"])
                        index_path = root_dir + "{}/{}/{}".format(folder, sub_folder, data[folder][sub_folder]["csv_2"])
                        bb_path = root_dir + "bbox/{}/{}/{}".format(folder, sub_folder, "c920-1.face")
                        #print(video_path, bb_path)
                        if "c920-1" not in video_path: #make sure bb_file fits video_file
                              video_path = root_dir + "{}/{}/{}".format(folder, sub_folder, data[folder][sub_folder]["video_2"])
                        if helper_functions.check_files_exist(video_path, ecg_path, index_path, bb_path):
                              video_paths.append(video_path)
                              ecg_paths.append(ecg_path)
                              idx_paths.append(index_path)
                              bb_data_paths.append(bb_path)
                        else:
                              continue
            return video_paths, ecg_paths, idx_paths, bb_data_paths
      
      def smooth_ecg(ecg, idx):
            #ecg[" ECG"] = helper_functions.detrend_ecg(ecg[" ECG HR"]) #detrend the signal
            
            #ecg["ECG_norm"] = (ecg[" ECG"] - ecg[" ECG"].mean()) / ecg[" ECG"].std() #Normalise
            ecg["ECG_norm"] = (ecg[" ECG"] - np.min(ecg[" ECG"] )) / (np.max(ecg[" ECG"] ) - np.min(ecg[" ECG"] ))
            ecg["ECG_MV"] = helper_functions.moving_average(ecg["ECG_norm"], 5) #moving average
            ecg = ecg.iloc[int(idx.iloc[0].idx_sig/5):] # start at the first frame of video
            return ecg
      
      def detrend_ecg(ecg_signal, smoothing_parameter=300):
            return np.convolve(ecg_signal, np.ones(smoothing_parameter) / smoothing_parameter, mode='same')
      
      def binary_mask(mask_array):
            mask_array = np.clip(mask_array[1:], 0, 1)
            return mask_array
      
      def get_video_frame_count(video_path):
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return frame_count
      
      def get_total_frame_count(video_paths):
            total_frames = 0
            for video_path in video_paths:
                  total_frames += helper_functions.get_video_frame_count(video_path)
            return total_frames
      
      def tensor_transform(mask_array, frame_array, ecg, frames):
            """
                  Function to transform to tensor. 
            """
            skin_seg_label = torch.tensor(np.array(mask_array))

            frame_tensor = torch.tensor(np.array(frame_array[1:]))
            frame_tensor = torch.swapaxes(frame_tensor, 0, 3)
            frame_tensor = torch.swapaxes(frame_tensor, 1, 2)
            frame_tensor = torch.swapaxes(frame_tensor, 1, 3)
            #if purpose == "val":
              #    ecg = ecg[1:frames+1].values
            #else:
            ecg = ecg["ECG_MV"][1:frames+1].values #only choose moving average and the amount of frames
            ecg = torch.tensor(np.array(ecg))
            return skin_seg_label, frame_tensor, ecg