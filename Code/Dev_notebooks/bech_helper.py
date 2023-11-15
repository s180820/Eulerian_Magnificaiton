import pandas as pd
import os
import cv2
from tqdm import tqdm
import numpy as np
import csv
# Function to process and convert .fys files to .csv
def convert_fys_to_csv(file_path):
    try:
        # Read the .fys file and skip the header line
        with open(file_path, "r") as f:
            lines = f.readlines()[1:]

        # Split the header to get column names
        header = "SampleNBR TimeStamp ECG-A(mV) MARKER(-) SAW(-)\n"

        # Create a list of lists to store the data
        data = []
        for line in lines:
            data.append(line.split())

        # Create a DataFrame using pandas
        df = pd.DataFrame(data, columns=header.strip().split())

        # Convert numerical columns to appropriate data types (e.g., float)
        df["SampleNBR"] = df["SampleNBR"].astype(int)
        df["TimeStamp"] = df["TimeStamp"].astype(float)
        df["ECG-A(mV)"] = df["ECG-A(mV)"].astype(float)
        df["MARKER(-)"] = df["MARKER(-)"].astype(int)
        df["SAW(-)"] = df["SAW(-)"].astype(int)

        # Save the DataFrame to a CSV file with the same name and .csv extension
        csv_file = os.path.splitext(file_path)[0] + ".csv"
        df.to_csv(csv_file, index=False)
        f.close()
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def converter_driver(dataset_dir):
    # Walk through the directory and its subfolders
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".fys"):
                print(file)
                file_path = os.path.join(root, file)
                convert_fys_to_csv(file_path)


def start_video_feed(video, network, conf=0.5):
    cap = cv2.VideoCapture(video)

    video_folder = os.path.dirname(video)
    output_csv = os.path.join(video_folder, "bbox.csv")

    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["Frame", "Confidence", "X", "Y", "Width", "Height"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        frame_count = 0

        frame_iterator = tqdm(
            enumerate(iter_frames(cap)),
            desc="Processing frames",
            total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

        for frame_count, frame in frame_iterator:
            # Grab dimensions to blob
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
            )

            # Pass blob through the network to perform facial detection
            network.setInput(blob)
            detections = network.forward()
            count = 0

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence < conf:
                    continue

                count += 1

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                width = endX - startX
                height = endY - startY

                # Write bbox information to the CSV file
                writer.writerow(
                    {
                        "Frame": frame_count,
                        "Confidence": confidence,
                        "X": startX,
                        "Y": startY,
                        "Width": width,
                        "Height": height,
                    }
                )

    cap.release()


def iter_frames(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame