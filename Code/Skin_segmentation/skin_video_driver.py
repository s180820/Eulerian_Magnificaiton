# Imports
print("[Video Driver] Loading imports and setting up enviroment...")
import pandas as pd
import numpy as np
from Skindetector import SkinDetector
import cv2
from tqdm import tqdm
import torch
import os

print("[Video Driver] Running...")

class MultipleVideoDriver():
    _class = "[Video Driver]"
    def convert_and_save_tensors(mask_array, frame_array, output_dir = None, saveTensors = False, verbosity = False):
        """
        Saving arrays as tensors. 
        """
        mask_tensor = torch.tensor(np.array(mask_array))
        frame_tensor = torch.tensor(np.array(frame_array))
        if verbosity:
            print(f"{MultipleVideoDriver._class} {frame_tensor.shape}")
        if saveTensors: 
            torch.save(mask_tensor, output_dir + 'mask_tensor.pt')
            torch.save(frame_tensor, output_dir + 'frame_tensor.pt')
            print("Saved tensors to disk. ")


    def convert_video_with_progress(video_file, data, starting_frame, frames_to_process = 64, output_file = None, video_size = 128, mask_size = 64, verbosity = True):
        mask_array = []
        frame_array = []
        video = cv2.VideoCapture(video_file)
        # Set starting frame
        video.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)

        cv2_total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        # Get total amount of frames
        if cv2_total_frames < frames_to_process:
            print(f"{MultipleVideoDriver._class} Requested frames are longer than the video - Setting max frames to vid size. ")
            frames_to_process = cv2_total_frames

        # Video capture flag.
        if output_file is not None: 
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file, fourcc, 30, (video_size, video_size))

        i = starting_frame # Adjust data to starting frame

        if verbosity:
            with tqdm(total=frames_to_process, unit="frames") as pbar:
                try:
                    frame_counter = 0
                    print(f"{MultipleVideoDriver._class} Reading and converting video...")
                    while True:
                        ret, frame = video.read()
                        if frame_counter == frames_to_process: 
                            break
                        if not ret:
                            print(f"{MultipleVideoDriver._class} Video ended")
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
                        mask_array.append(np.array(mask, dtype='float32'))
                        frame_array.append(np.array(frame, dtype='float32'))

                        # Save the processed frame to the output video
                        if output_file is not None:
                            out.write(skin)
                        pbar.update(1)

                        _, frame = cv2.imencode('.jpeg', skin)
                        frame_counter += 1
                except KeyboardInterrupt:
                    pass
                finally:
                    video.release()
                    if output_file is not None:
                        out.release()
                    return mask_array, frame_array
        else:
            try:
                frame_counter = 0
                while True:
                    ret, frame = video.read()
                    if frame_counter == frames_to_process: 
                        break
                    if not ret:
                        print(f"{MultipleVideoDriver._class} Video ended.")
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
                    mask_array.append(np.array(mask, dtype='float32'))
                    frame_array.append(np.array(frame, dtype='float32'))
                    # Save the processed frame to the output video
                    if output_file is not None:
                         out.write(skin)

                    _, frame = cv2.imencode('.jpeg', skin)
                    frame_counter += 1
            except KeyboardInterrupt:
                    pass
            finally:
                video.release()
                if output_file is not None:
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
    MultipleVideoDriver.create_directory_if_not_exists(output_dir) 

    data = pd.read_csv(bb_file, sep=" ", header=None, names=["frame", "x", "y", "w", "h"]).drop("frame", axis=1)
    output_file = os.path.join(output_dir, 'output_video.mp4')

    # Usage
    mask_array, frame_array = MultipleVideoDriver.convert_video_with_progress(video_file = video_file, data = data, output_file = output_file, 
                                                                              frames_to_process=50000,
                                                                              starting_frame=1, verbosity=True)

    # Convert and save tensors
    MultipleVideoDriver.convert_and_save_tensors(mask_array, frame_array, output_dir = output_dir, saveTensors=True)