# source https://github.com/gopinath-balu/computer_vision/tree/master
import numpy as np
import argparse
import time
import cv2

import tkinter as tk
from tkinter import filedialog


def start_live_feed(cap, network):
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (500, 350))

        # Grap dimensions to blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(
            frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        # Pass blot through network to perform facial detection
        network.setInput(blob)
        detections = network.forward()
        count = 0

        for i in range(0, detections.shape[2]):
            # Extract confidence assoficated with predictions.
            confidence = detections[0, 0, i, 2]

            # Filter based on confidence
            if confidence < arguments["confidence"]:
                continue
            count += 1

            # compute BBOX
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw box
            text = "{:.2f}%".format(confidence * 100) + \
                ", Count: " + str(count)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY),
                          (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    cap.release()


def start_video_feed(video, network):
    cap = cv2.VideoCapture(video)
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (500, 350))

        # Grap dimensions to blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(
            frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        # Pass blot through network to perform facial detection
        network.setInput(blob)
        detections = network.forward()
        count = 0

        for i in range(0, detections.shape[2]):
            # Extract confidence assoficated with predictions.
            confidence = detections[0, 0, i, 2]

            # Filter based on confidence
            if confidence < arguments["confidence"]:
                continue
            count += 1

            # compute BBOX
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw box
            text = "{:.2f}%".format(confidence * 100) + \
                ", Count: " + str(count)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY),
                          (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    cap.release()


def chosefile():
    filetypes = (
        ('MP4', '*.mp4'),
        ('All files', '*.*'),
    )

    root = tk.Tk()
    root.overrideredirect(True)  # hides the window
    # sets the window transparency to 0 (invisible)
    root.attributes("-alpha", 0)
    filename = tk.filedialog.askopenfilename(
        title='Select a file...',
        filetypes=filetypes,
    )
    root.destroy()
    return filename


def startup(arguments):
    print("Loading model from disk...")
    # Load models
    network = cv2.dnn.readNetFromCaffe(
        arguments.get("prototxt"), arguments.get("model"))
    choise = input("Do you wanna use real time video? (y/n): ")
    # Initialize video stream (from webcam)
    if choise == "y":
        print("[INFO] starting video stream from build-in webcam")
        print("[CMD] press 'q' to exit")
        cap = cv2.VideoCapture(0)
        start_live_feed(cap, network=network)
    else:
        print(["[INFO] select video file.."])
        filename = chosefile()
        print("[INFO] starting video stream from file: " + filename)
        print("[CMD] press 'q' to exit")
        start_video_feed(network=network, video=filename)
        # open file expolorer to locate video file


# main
if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("-p", "--prototxt",
                           help="path to Caffe 'deploy' prototxt (TXT format) file", default="models/deploy.prototxt.txt")
    arg.add_argument("-m", "--model", help="path to Caffe pre-trained model",
                           default="models/Facial_recognition/res10_300x300_ssd_iter_140000.caffemodel")
    arg.add_argument("-c", "--confidence", type=float, default=0.5,
                           help="minimum probability to filter weak detections")
    arguments = vars(arg.parse_args())
    startup(arguments)
