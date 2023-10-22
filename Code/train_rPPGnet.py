###   it is just for research purpose, and commercial use is not allowed  ###

import torch
import json
import pandas as pd
import numpy as np
import sys
import os
import wandb
import pickle
from scipy import signal

sys.path.append('Code/Skin_segmentation')
sys.path.append('models')

import skin_detection_runfile
from rPPGNet import *

from wandb_config import wandb_defaults

### WANDB
wandb_settings = wandb_defaults.copy()
wandb_settings.update({"dir" : "logs/" + wandb_settings.get("project") + "/" + "Eulerian"}) # create log dir
wandb_settings.update({"name" : "Eulerian", "group": "rPPGNet"}) # set run name

# create directory for logs if first run in project
os.makedirs(wandb_settings["dir"], exist_ok=True)

# init wandb
wandb_run = wandb.init(**wandb_settings)

def clear_cache():
        os.system("rm -rf ~/.cache/wandb")

'''  ###############################################################
#
#   Step 1:  two loss function
#          1.1   nn.BCELoss()  for skin segmentation
#          1.2   Neg_Pearson()  for rPPG signal regression
#
'''  ###############################################################

class Neg_Pearson(torch.nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Neg_Pearson,self).__init__()
        return
    def forward(self, preds, labels):       # all variable operation
        
        loss = 0
        
        sum_x = torch.sum(preds[0])                # x
        sum_y = torch.sum(labels)               # y
        sum_xy = torch.sum(preds[0]*labels)        # xy
        sum_x2 = torch.sum(torch.pow(preds[0],2))  # x^2
        sum_y2 = torch.sum(torch.pow(labels,2)) # y^2
        N = preds.shape[1]
        pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))

        #if (pearson>=0).data.cpu().numpy():    # torch.cuda.ByteTensor -->  numpy
            #   loss += 1 - pearson
        #else:
            #   loss += 1 - torch.abs(pearson)
        
        loss += 1 - pearson
            
            
        #loss = loss/preds.shape[0]
        return loss

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
    # Remove null entries from the list 
    #final_list = moving_averages_list[window_size - 1:] 
    
    return final_list

criterion_Binary = torch.nn.BCELoss()  # binary segmentation
criterion_Pearson = Neg_Pearson()   # rPPG singal 
criterion_MSE = torch.nn.MSELoss()


'''   ###############################################################
#
#   Step 2: Forward model and calculate the losses  
#           # inputs: facial frames --> [3, 64, 128, 128]
            # skin_seg_label: binary skin labels --> [64, 64, 64] 
            # ecg: groundtruth smoothed ecg signals --> [64]
#            
#            2.1  Forward the model, get the predicted skin maps and rPPG signals
#            2.2  Calculate the loss between predicted skin maps and binary skin labels (loss_binary)
#            2.3  Calculate the loss between predicted rPPG signals and groundtruth smoothed ecg signals (loss_ecg, loss_ecg1, loss_ecg2, loss_ecg3,## loss_ecg4, loss_ecg_aux)      
#
'''   ###############################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

no_of_frames = 64
model = rPPGNet(frames = no_of_frames)
wandb_run.watch(model)

#for name, param in model.named_parameters():
 #   if param.requires_grad:
  #      print(name)

rate_learning = 1e-4
optim = torch.optim.Adam(model.parameters(), lr=rate_learning)


# Load data
with open('Data/json_structure') as json_file:
    data = json.load(json_file)
i = 0
root_dir = "/work3/s174159/data/"
for folder in data:
    for sub_folder in data[folder]:
        video_path = root_dir + "{}/{}/{}".format(folder, sub_folder, data[folder][sub_folder]["video_1"])
        ecg_path = root_dir + "{}/{}/{}".format(folder, sub_folder, data[folder][sub_folder]["csv_1"])
        index_path = root_dir + "{}/{}/{}".format(folder, sub_folder, data[folder][sub_folder]["csv_2"])
        bb_file = root_dir + "bbox/{}/{}/{}".format(folder, sub_folder, "c920-1.face")
        bb_data = pd.read_csv(bb_file, sep=" ", header=None, names=["frame", "x", "y", "w", "h"]).drop("frame", axis=1)

        if "c920-1" not in video_path: #make sure bb_file fits video_file
            video_path = root_dir + "{}/{}/{}".format(folder, sub_folder, data[folder][sub_folder]["video_2"])

        print(video_path)
        print(bb_file)
        # Usage
        mask_array, frame_array = skin_detection_runfile.convert_video_with_progress(video_path, bb_data, frames = no_of_frames)
        #ecg = pd.read_csv(ecg_path)["ECG HR"].values[:no_of_frames]
        idx = pd.read_csv(index_path, header=None, names=["timestamp", "idx_sig"])
        ecg = pd.read_csv(ecg_path)

        #Smoothing ECG Signal
        ecg[" ECG"] = signal.detrend(ecg[" ECG HR"]) #detrending
        ecg["ECG_norm"] = (ecg[" ECG"] - ecg[" ECG"].mean()) / ecg[" ECG"].std() #normalizing
        ecg = ecg.iloc[::5, :] #choosing only the signal picked up by the vitacom
        ecg["ECG_MV"] = moving_average(ecg["ECG_norm"], 5) #calculate 5 point moving average
        
        ecg = ecg.iloc[int(idx.iloc[0].idx_sig/5):] #start the data at the start of the video
        print(ecg)
        #ecg = ecg.groupby(by="milliseconds").mean() #
        #mov_avg = moving_average(ecg["ECG_norm"], 5)[:no_of_frames]
        #ecg = ecg.iloc[idx.iloc[1].idx_sig:idx.iloc[-1].idx_sig + 4] # Getting the correct frames
        #ecg = ecg.groupby(by="milliseconds").mean()
        #mov_avg = mov_avg[idx.iloc[1].idx_sig:idx.iloc[-1].idx_sig + 4]
        ecg = ecg["ECG_MV"][1:no_of_frames+1].values #get only the first number of specified frames
        #print(ecg)
        #print(mov_avg)
        #print(len(mov_avg))

        assert len(ecg) == no_of_frames
        #print(ecg)
        # Convert and save tensors
        mask_array = np.clip(mask_array[1:], 0, 1)
        skin_seg_label = torch.tensor(np.array(mask_array)).unsqueeze(0)
        frame_tensor = torch.tensor(np.array(frame_array[1:]))
        frame_tensor = torch.swapaxes(frame_tensor, 0, 3)
        frame_tensor = torch.swapaxes(frame_tensor, 1, 2)
        frame_tensor = torch.swapaxes(frame_tensor, 1, 3)
        frame_tensor = frame_tensor.unsqueeze(0) # [1, 3, no_of_frames, 128, 128]
        	
        ecg = torch.tensor(np.array(ecg))
        #ecg = (ecg-torch.mean(ecg)) /torch.std(ecg) # normalisation
        #ecg = torch.tensor(median_filter(ecg, 3))
        #print(ecg)
        
        # ecg = torch.rand(no_of_frames)

        optim.zero_grad()
        
        skin_map, rPPG_aux, rPPG, rPPG_SA1, rPPG_SA2, rPPG_SA3, rPPG_SA4, x_visual6464, x_visual3232  = model(frame_tensor)
        rPPG = (rPPG-torch.mean(rPPG)) /torch.std(rPPG)	 	# normalize2
        rPPG_SA1 = (rPPG_SA1-torch.mean(rPPG_SA1)) /torch.std(rPPG_SA1)	 	# normalize2
        rPPG_SA2 = (rPPG_SA2-torch.mean(rPPG_SA2)) /torch.std(rPPG_SA2)	 	# normalize2
        rPPG_SA3 = (rPPG_SA3-torch.mean(rPPG_SA3)) /torch.std(rPPG_SA3)	 	# normalize2
        rPPG_SA4 = (rPPG_SA4-torch.mean(rPPG_SA4)) /torch.std(rPPG_SA4)	 	# normalize2
        rPPG_aux = (rPPG_aux-torch.mean(rPPG_aux)) /torch.std(rPPG_aux)	 	# normalize2
        
        with open('test.pickle', 'wb') as handle:
            pickle.dump((ecg, rPPG_aux, rPPG, rPPG_SA1, rPPG_SA2, rPPG_SA3, rPPG_SA4), handle)

        loss_binary = criterion_Binary(skin_map, skin_seg_label) 
        loss_ecg = criterion_Pearson(rPPG, ecg)
        loss_ecg1 = criterion_Pearson(rPPG_SA1, ecg)
        loss_ecg2 = criterion_Pearson(rPPG_SA2, ecg)
        loss_ecg3 = criterion_Pearson(rPPG_SA3, ecg)
        loss_ecg4 = criterion_Pearson(rPPG_SA4, ecg)
        loss_ecg_aux = criterion_Pearson(rPPG_aux, ecg)
        
        MSE_err = criterion_MSE(rPPG[0], ecg)

        '''   ###############################################################
        #
        #   Step 3:  loss fusion and BP  
        #
        '''   ###############################################################

        loss = 0.1*loss_binary +  0.5*(loss_ecg1 + loss_ecg2 + loss_ecg3 + loss_ecg4 + loss_ecg_aux) + loss_ecg
        wandb_run.log({
            "Train metrics/Binary_Loss":  loss_binary,
            "Train metrics/ecg_Loss":  loss_ecg,
            "Train metrics/ecg1_Loss":  loss_ecg1,
            "Train metrics/ecg2_Loss":  loss_ecg2,
            "Train metrics/ecg3_Loss":  loss_ecg3,
            "Train metrics/ecg4_Loss":  loss_ecg4,
            "Train metrics/ecg_aux_Loss":  loss_ecg_aux,
            "Train metrics/Total_Loss":  loss,
            "Train metrics/RMSE": torch.sqrt(MSE_err),
            "video":  video_path,
                })
        print("Loss for iteration", i, ":", loss.item())
        print("Loss for ecg1: ", loss_ecg1.item())
        print("Loss for ecg2: ", loss_ecg2.item())
        print("Loss for ecg3: ", loss_ecg3.item())
        print("Loss for ecg4: ", loss_ecg4.item())
        print("RMSE: ", torch.sqrt(MSE_err).item())
        i += 1
            # clear cache
        clear_cache()

        loss.backward()
        optim.step()

wandb.finish()