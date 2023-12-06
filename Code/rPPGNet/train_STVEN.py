###   it is just for research purpose, and commercial use is not allowed  ###

import torch
from models.STVEN import *
import os
import torchvision
import subprocess

from compress import extract_bitrate

'''  ###############################################################
#
#   Step 1: loss functions for STVEN 
#          1.1   torch.mean(torch.abs(x_reconst - video_y_GT))  for L1 reconstruction loss
#          1.2   psnr()  for L2 reconstruction loss
#          1.3   torch.mean(torch.abs(x_reconst -video_x_GT)) Cycle reconstruction loss
#
'''  ###############################################################

def psnr(self, img, img_g):
        
        criterionMSE = nn.MSELoss() #.to(device)
        mse = criterionMSE(img, img_g)
        psnr = 10 * torch.log10(1./ (mse+10e-8)) #20 *

        return  psnr



'''   ###############################################################
#
#   Step 2: Forward model and calculate the losses  
#           # input 1: facial frames --> [3, 64, 128, 128]
            # input 2: target label mask --> 5D vector 
#			# video_GroudTruth: the original video (before highly compressed)
#            
#            2.1  Forward the model, Generate video from original video to the target video
#            2.2  Calculate the reconstruction loss
#            2.3  Calculate the PSNR loss
#            2.4  Calculate the cycle loss       
#
'''   ###############################################################


model = STVEN_Generator()

avg_loss = 0

gt_dir = 'folketinget/clips'
gt_list = os.listdir(gt_dir)
gt_list.sort()
gt_list = [os.path.join(gt_dir, x) for x in gt_list]

compressed_dir = 'folketinget/compressed'
compressed_list = os.listdir(compressed_dir)
compressed_list.sort()
compressed_list = [os.path.join(compressed_dir, x) for x in compressed_list]

for i in range(100):
    print(compressed_list[i])
    # load target label mask
    traget_label1 = extract_bitrate(compressed_list[i])
    traget_label1 = torch.tensor(traget_label1)
    # load orginal label mask
    original_label1 = extract_bitrate(gt_list[i])
    original_label1 = torch.tensor(original_label1)
    print(traget_label1.view(-1))
    print(original_label1.view(-1))
    # load compressed video from mp4 file
    comp_vid = torchvision.io.read_video(compressed_list[i])
    video_1 = comp_vid

    
    video_1 = []
    for video_frame in comp_vid:
        video_1.append(video_frame)
    print(video_1[0].size())
    
    gt_vid = torchvision.io.read_video(compressed_list[i])
    video_GroudTruth = gt_vid[0]

    
    #video_GroudTruth = torch.load(gt_list[i])
    x_reconst = model(video_1, traget_label1)

    L1_loss = torch.mean(torch.abs(x_reconst - video_GroudTruth))
    Loss_PSNR = psnr(x_reconst, video_GroudTruth)

    x_fake = model(x_reconst, original_label1)

    L1_loss_cycle = torch.mean(torch.abs(x_fake - video_1))
    Loss_PSNR_cycle = psnr(x_fake, video_1)  

    loss = 100*L1_loss + Loss_PSNR+ 100*L1_loss_cycle + Loss_PSNR_cycle
    print("Loss for iteration", i, ":", loss.item())
    avg_loss += loss.item()
    loss.backward()


'''   ###############################################################
#
#   Step 3:  loss fusion and BP  
#
'''   ###############################################################

avg_loss = avg_loss / 100