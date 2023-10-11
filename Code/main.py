from custom_dataloader import CustomDataset
from torch.utils.data import DataLoader

print("Starting....")
# Define the root directory of your dataset and resize shape
dataset_root = '/work3/s174159/data'
resize_shape = (320, 240)  # Adjust the resolution as needed

# Create an instance of the CustomDataset
custom_dataset = CustomDataset(dataset_root, resize_shape=resize_shape)
print("Initiated custom dataset ....")
# Create a DataLoader to load batches of data

print("Doing the data loader")
batch_size = 2
dataloader = DataLoader(custom_dataset, batch_size=2, shuffle=True)

print("Dataloader created....")


for batch in dataloader:
    print("Doing batches")
    video_frames_batch = batch['video_frames']

    csv_data_batch = batch['csv_data']
