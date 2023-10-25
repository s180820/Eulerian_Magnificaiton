from scipy import signal
import pandas as pdf
import numpy as np
import torch

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

      def smooth_ecg(ecg, idx, purpose):
            ecg[" ECG"] = signal.detrend(ecg[" ECG HR"])
            ecg["ECG_norm"] = ecg[" ECG"] - ecg[" ECG"].mean() / ecg[" ECG"].std() # Normalize
            ecg = ecg.groupby(by="milliseconds").mean()
            ecg["ECG_MV"] = helper_functions.moving_average(ecg["ECG_norm"], 5)
            ecg = ecg.iloc[int(idx.iloc[0].idx_sig/5):]
            return ecg
      
      def binary_mask(mask_array):
            mask_array = np.clip(mask_array[1:], 0, 1)
            return mask_array
      
      def tensor_transform(mask_array, frame_array, ecg, frames, purpose):
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