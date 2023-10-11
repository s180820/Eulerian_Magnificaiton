import glob
from IPython.display import Video, HTML
import os
import cv2
import pandas as pd
from torch.utils.data import Dataset
import torch


class CustomDataset(Dataset):
    def __init__(self, root_dir, resize_shape=(320, 240)):  # Adjust the resolution as needed
        self.root_dir = root_dir
        self.resize_shape = resize_shape
        self.samples = []

        # Traverse the directory structure
        for dirpath, _, filenames in os.walk(root_dir):
            if any(file.endswith('.avi') for file in filenames):
                # Check if there are video files in this directory
                csv_file = os.path.join(dirpath, 'viatom-raw.csv')
                video_files = [file for file in filenames if file.endswith('.avi')]
                for video_file in video_files:
                    video_path = os.path.join(dirpath, video_file)
                    self.samples.append((video_path, csv_file))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, csv_path = self.samples[idx]

        # Load video frames and resize
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Resize frame
            frame = cv2.resize(frame, self.resize_shape)
            # You may want to further preprocess the frame here
            frames.append(frame)

        cap.release()
        video_frames = torch.tensor(frames)

        # Load CSV data
        csv_data = pd.read_csv(csv_path)

        return {
            'video_frames': video_frames,
            'csv_data': csv_data,
        }



