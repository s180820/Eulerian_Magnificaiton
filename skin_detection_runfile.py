# Imports
print("Loading imports and setting up enviroment...")
import pandas as pd
import numpy as np
from Skindetector import SkinDetector
import cv2
from tqdm import tqdm
import torch
print("Finished...")


def convert_and_save_tensors(mask_array, frame_array):
    """
    Saving arrays as tensors. 
    """
    mask_tensor = torch.tensor(np.array(mask_array))
    frame_tensor = torch.tensor(np.array(frame_array))
    torch.save(mask_tensor, 'mask_tensor.pt')
    torch.save(frame_tensor, 'frame_tensor.pt')
    
    print("Saved tensors to disk. ")


def convert_video_with_progress(video_file, data, output_file, video_size = 128, mask_size = 64):
    mask_array = []
    frame_array = []
    video = cv2.VideoCapture(video_file)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 30, (video_size, video_size))
    
    i = 0

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with tqdm(total=total_frames, unit="frames") as pbar:
        try:
            print("Reading and converting video...")
            while True:
                ret, frame = video.read()
                if not ret:
                    print("Finished video...")
                    break
                
                if i == len(data):
                    i = 0
                x, y, w, h = data.iloc[i].values.astype(int)
                i += 1
                frame = frame[y:y + h, x:x + w]
                frame = cv2.resize(frame, (video_size, video_size), interpolation=cv2.INTER_AREA)

                detector = SkinDetector(frame)  # You need to define the SkinDetector class
                detector.find_skin()
                image, mask, skin = detector.get_resulting_images()
                mask = cv2.resize(mask, (mask_size, mask_size), interpolation=cv2.INTER_AREA)

                # Add mask, frame to array
                mask_array.append(mask)
                frame_array.append(frame)


                
                # Save the processed frame to the output video
                out.write(skin)
                pbar.update(1)
                
                _, frame = cv2.imencode('.jpeg', skin)
        except KeyboardInterrupt:
            pass
        finally:
            video.release()
            out.release()
            return mask_array, frame_array



# Set directories: 
root_dir = "/work3/s174159/data/"
bb_file = root_dir + "bbox/00/01/c920-1.face"
video_file = root_dir + "00/01/c920-1.avi"
data = pd.read_csv(bb_file, sep=" ", header=None, names=["frame", "x", "y", "w", "h"]).drop("frame", axis=1)
output_file = 'output_video.mp4'

# Usage
mask_array, frame_array = convert_video_with_progress(video_file, data, output_file)

# Convert and save tensors
convert_and_save_tensors(mask_array, frame_array)