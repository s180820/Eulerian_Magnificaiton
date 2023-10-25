import os
from torch.utils.data import Dataset
from torchvision import transforms
import json

class CustomDataset(Dataset):
    def __init__(self, root_dir, output_json_file, resize_shape=(320, 240)):  # Adjust the resolution as needed
        self.root_dir = root_dir
        self.resize_shape = resize_shape
        self.samples = []
        self.output_json_file = output_json_file
        self.structure = {}
        self.transform = transforms.Compose([transforms.ToTensor()])

        # Traverse the directory structure
        for dirpath, _, filenames in os.walk(root_dir):
            if any(file.endswith('.avi') for file in filenames):
                # Check if there are video files in this directory
                csv_file = os.path.join(dirpath, 'viatom-raw.csv')
                video_files = [file for file in filenames if file.endswith('.avi')]
                for video_file in video_files:
                    video_path = os.path.join(dirpath, video_file)
                    self.samples.append((video_path, csv_file))

    def create_directory_structure(self):
        structure = {}
        for root, dirs, files in os.walk(self.root_dir):
            current_level = structure
            folders = os.path.relpath(root, self.root_dir).split(os.sep)
            dont_matter = ["bbox", ".", "ForPublication"]
            for folder in folders:
                if folder in dont_matter:
                    continue
                current_level = current_level.setdefault(folder, {})
            video_files = [file for file in files if file.endswith('.avi')]
            csv_files = [file for file in files if file.endswith('.csv')]
            try:
                current_level.update({"video_1" : video_files[0],
                                    "video_2" : video_files[1],
                                    "csv_1" : csv_files[0],
                                    "csv_2" : csv_files[1],})
            except IndexError:
                continue
        
        return structure

    def save_structure_to_json(self):
        with open(self.output_json_file, 'w') as f:
            json.dump(self.structure, f, indent=4)

    def __len__(self):
        return len(self.samples)
    
    def __iter__(self):
        self._iter_structure = iter(self.structure)
        return self

    def __next__(self):
        try:
            key = next(self._iter_structure)
            return key, self.structure[key]
        except StopIteration:
            raise StopIteration

    def __getitem__(self, idx):
        #video_path, csv_path = self.samples[idx]
        self.structure = self.create_directory_structure()
        self.save_structure_to_json()
        