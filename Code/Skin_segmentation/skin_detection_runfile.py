# Imports
print("Loading imports and setting up enviroment...")
import pandas as pd
import numpy as np
from Skindetector import SkinDetector
import cv2
from tqdm import tqdm
import torch
import os
print("Finished...")


def convert_and_save_tensors(mask_array, frame_array, output_dir = None, saveTensors = False):
    """
    Saving arrays as tensors. 
    """
    mask_tensor = torch.tensor(np.array(mask_array))
    frame_tensor = torch.tensor(np.array(frame_array))
    print(frame_tensor.shape)
    if saveTensors: 
        torch.save(mask_tensor, output_dir + 'mask_tensor.pt')
        torch.save(frame_tensor, output_dir + 'frame_tensor.pt')
        print("Saved tensors to disk. ")


def convert_video_with_progress(video_file, data, output_file = None, video_size = 128, mask_size = 64, outputvideo = False):
    mask_array = []
    frame_array = []
    video = cv2.VideoCapture(video_file)
    
    # Video capture flag.
    if outputvideo: 
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, 30, (video_size, video_size))
    
    i = 0

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(total_frames)
    
    with tqdm(total=total_frames, unit="frames") as pbar:
        try:
            print("Reading and converting video...")
            while True:
                ret, frame = video.read()
                print("ret:", ret)
                if not ret:
                    print("Finished video...")
                    break
                
                # if i == len(data):
                #     i = 0
                x, y, w, h = data.iloc[i].values.astype(int)
                i += 1
                frame = frame[y:y + h, x:x + w]
                frame = cv2.resize(frame, (video_size, video_size), interpolation=cv2.INTER_AREA)

                detector = SkinDetector(frame)  # You need to define the SkinDetector class
                detector.find_skin()
                _, mask, skin = detector.get_resulting_images()
                mask = cv2.resize(mask, (mask_size, mask_size), interpolation=cv2.INTER_AREA)

                # Add mask, frame to array
                mask_array.append(mask)
                frame_array.append(frame)
                print(len(frame_array))
                
                # Save the processed frame to the output video
                if outputvideo:
                    out.write(skin)
                pbar.update(1)
                
                _, frame = cv2.imencode('.jpeg', skin)
        except KeyboardInterrupt:
            pass
        finally:
            video.release()
            if outputvideo:
                out.release()
            return mask_array, frame_array

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Set directories: 
if __name__ == "__main__":
    root_dir = "/work3/s174159/data/"
    bb_file = root_dir + "bbox/00/01/c920-1.face"
    video_file = root_dir + "00/01/c920-1.avi"

    output_dir = '/zhome/01/d/127159/Desktop/Eulerian_Magnificaiton/output_dir/Skin_segmentation/'
    create_directory_if_not_exists(output_dir) 

    data = pd.read_csv(bb_file, sep=" ", header=None, names=["frame", "x", "y", "w", "h"]).drop("frame", axis=1)
    output_file = os.path.join(output_dir, 'output_video.mp4')

    # Usage
    mask_array, frame_array = convert_video_with_progress(video_file, data, output_file, outputvideo=False)

    # Convert and save tensors
    convert_and_save_tensors(mask_array, frame_array, output_dir = output_dir, saveTensors=False)
