###   it is just for research purpose, and commercial use is not allowed  ###

import torch
import json
import pandas as pd
import numpy as np
import sys
import os
import wandb

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
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])                # x
            sum_y = torch.sum(labels[i])               # y
            sum_xy = torch.sum(preds[i]*labels[i])        # xy
            sum_x2 = torch.sum(torch.pow(preds[i],2))  # x^2
            sum_y2 = torch.sum(torch.pow(labels[i],2)) # y^2
            N = preds.shape[1]
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))

            #if (pearson>=0).data.cpu().numpy():    # torch.cuda.ByteTensor -->  numpy
            #    loss += 1 - pearson
            #else:
            #    loss += 1 - torch.abs(pearson)
            
            loss += 1 - pearson
            
            
        loss = loss/preds.shape[0]
        return loss



criterion_Binary = torch.nn.BCELoss()  # binary segmentation
criterion_Pearson = Neg_Pearson()   # rPPG singal 


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



no_of_frames = 256
model = rPPGNet(frames = no_of_frames)
wandb_run.watch(model)

# Load data
with open('Data/json_structure') as json_file:
    data = json.load(json_file)

root_dir = "/work3/s174159/data/"
for folder in data:
    for sub_folder in data[folder]:
        video_path = root_dir + "{}/{}/{}".format(folder, sub_folder, data[folder][sub_folder]["video_1"])
        ecg_path = root_dir + "{}/{}/{}".format(folder, sub_folder, data[folder][sub_folder]["csv_2"])
        bb_file = root_dir + "bbox/{}/{}/{}".format(folder, sub_folder, "c920-1.face")

        bb_data = pd.read_csv(bb_file, sep=" ", header=None, names=["frame", "x", "y", "w", "h"]).drop("frame", axis=1)

        # Usage
        mask_array, frame_array = skin_detection_runfile.convert_video_with_progress(video_path, bb_data, frames = no_of_frames)
        ecg = pd.read_csv(ecg_path, header=None)[1].values
        # Convert and save tensors
        mask_array = np.clip(mask_array, 0, 1)
        skin_seg_label = torch.tensor(np.array(mask_array)).unsqueeze(0)
        frame_tensor = torch.tensor(np.array(frame_array))
        frame_tensor = torch.swapaxes(frame_tensor, 0, 3)
        frame_tensor = torch.swapaxes(frame_tensor, 1, 3)
        frame_tensor = frame_tensor.unsqueeze(0)
        ecg = torch.tensor(np.array(ecg))

        skin_map, rPPG_aux, rPPG, rPPG_SA1, rPPG_SA2, rPPG_SA3, rPPG_SA4, x_visual6464, x_visual3232  = model(frame_tensor)


        loss_binary = criterion_Binary(skin_map, skin_seg_label)  

        rPPG = (rPPG-torch.mean(rPPG)) /torch.std(rPPG)	 	# normalize2
        rPPG_SA1 = (rPPG_SA1-torch.mean(rPPG_SA1)) /torch.std(rPPG_SA1)	 	# normalize2
        rPPG_SA2 = (rPPG_SA2-torch.mean(rPPG_SA2)) /torch.std(rPPG_SA2)	 	# normalize2
        rPPG_SA3 = (rPPG_SA3-torch.mean(rPPG_SA3)) /torch.std(rPPG_SA3)	 	# normalize2
        rPPG_SA4 = (rPPG_SA4-torch.mean(rPPG_SA4)) /torch.std(rPPG_SA4)	 	# normalize2
        rPPG_aux = (rPPG_aux-torch.mean(rPPG_aux)) /torch.std(rPPG_aux)	 	# normalize2

        loss_ecg = criterion_Pearson(rPPG, ecg)
        loss_ecg1 = criterion_Pearson(rPPG_SA1, ecg)
        loss_ecg2 = criterion_Pearson(rPPG_SA2, ecg)
        loss_ecg3 = criterion_Pearson(rPPG_SA3, ecg)
        loss_ecg4 = criterion_Pearson(rPPG_SA4, ecg)
        loss_ecg_aux = criterion_Pearson(rPPG_aux, ecg)


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
            "video":  video_path,
                })
            
            # clear cache
        clear_cache()

        loss.backward()

wandb.finish()