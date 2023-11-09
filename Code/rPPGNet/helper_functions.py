from scipy import signal
import pandas as pdf
import numpy as np
import torch
import cv2
import json
import os
import scipy.fftpack as fftpack
from scipy.signal import butter, lfilter

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
      
      def bandpass_filter(data, lowcut = 0.05, highcut = 100.0, fs = 500, order=2):
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(order, [low, high], btype="band")
            y = lfilter(b, a, data)
            return y
      
      def smooth_ecg(ecg, idx):
            #ecg[" ECG"] = helper_functions.detrend_ecg(ecg[" ECG HR"]) #detrend the signal
            lead_names = ["Lead I", "Lead II", "Lead III", "Lead aVR", "Lead aVL"]
            ecg["Lead"] = lead_names * (len(ecg) // 5)
            ecg["time"] = (ecg["milliseconds"] - ecg["milliseconds"].min()) / 1000  # Convert time.
            ecg = ecg.loc[idx.iloc[0].idx_sig:idx.iloc[-1].idx_sig+1]
            #print(ecg)
            true_ecg = ecg[ecg["Lead"] == "Lead II"][" ECG"].reset_index(drop=True) - ecg[ecg["Lead"] == "Lead I"][" ECG"].reset_index(drop=True)
            ecg_norm = (true_ecg - np.mean(true_ecg)) / np.std(true_ecg) # GUSTAV 
            #ecg_mv = helper_functions.moving_average(ecg_norm, 3) #moving average # GUSSE
            ecg_mv = helper_functions.bandpass_filter(ecg_norm)
             # start at the first frame of video
            return ecg_mv
      
      def fft_filter(video, freq_min, freq_max, fps):
            fft = fftpack.fft(video, axis=0)
            frequencies = fftpack.fftfreq(video.shape[0], d=1.0 / fps)
            bound_low = (np.abs(frequencies - freq_min)).argmin()
            bound_high = (np.abs(frequencies - freq_max)).argmin()
            fft[:bound_low] = 0
            fft[bound_high:-bound_high] = 0
            fft[-bound_low:] = 0
            iff = fftpack.ifft(fft, axis=0)
            result = np.abs(iff)
            result *= 100  # Amplification factor

            return result, fft, frequencies

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
            # frame_tensor = torch.swapaxes(frame_tensor, 2, 3)
            # frame_tensor = torch.swapaxes(frame_tensor, 0, 1)
            #if purpose == "val":
              #    ecg = ecg[1:frames+1].values
            #else:
            ecg = np.abs(ecg[1:frames+1]) #only choose moving average and the amount of frames
            ecg = torch.tensor(np.array(ecg))
            return skin_seg_label, frame_tensor, ecg