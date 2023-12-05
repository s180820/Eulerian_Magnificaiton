import numpy as np
import pandas as pd
import plotly.express as px
import os
import cv2 
from main import *
from itertools import cycle
from plotly.validators.scatter.marker import SymbolValidator
from sklearn.metrics import mean_absolute_error

def buildGauss(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid

def reconstructFrame(pyramid, index, levels, videoWidth=160, videoHeight=120):
    filteredFrame = pyramid[index]
    for level in range(levels):
        filteredFrame = cv2.pyrUp(filteredFrame)
    filteredFrame = filteredFrame[:videoHeight, :videoWidth]
    return filteredFrame

def convert_avi_to_mp4(avi_file_path, output_name):
    os.popen("ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(input = avi_file_path, output = output_name))
    return True

PROTOTXT_PATH = "../../Models/Facial_recognition/deploy_prototxt.txt"
MODEL_PATH = "../../Models/Facial_recognition/res10_300x300_ssd_iter_140000.caffemodel"

#dataloader from val_data folder
video_list = []
label_list = []

for folder in os.listdir("val_data"):
    for file in os.listdir("val_data/"+ folder):
        if file.endswith(".mp4"):
            #convert avi to mp4
            #convert_avi_to_mp4(os.path.join("val_data/"+folder, file), os.path.join("val_data/"+folder, file[:-4]))
            video_list.append(os.path.join("val_data/"+folder, file))
        elif file.endswith(".rr"):
            label_list.append(os.path.join("val_data/"+folder, file))

#sort lists
video_list.sort()
label_list.sort()

mae_hr_cnn_list = []
mae_mtts_can_list = []
mae_eulerian_list = []

for i in range(len(video_list)):
    rr = pd.read_csv(label_list[i], header=None)
    #split data column into two columns
    rr = rr[0].str.split(' ', expand=True)
    rr = rr.rename(columns={0:'time', 1:'rr'})
    hr = 60/rr['rr'].astype(float)
    # repeat "Ground Truth" for 60 times
    gt = np.repeat("Ground Truth", 60)

    # create dataframe for plot 
    data = np.array([range(1, 61), hr[:60].values, gt]).T
    #plot_df = pd.DataFrame([*range(1, 61), hr[:60].values, gt], columns=["Time", "BPM", "type"])
    plot_df = pd.DataFrame(data = data, columns=["Time", "BPM", "type"])

    # add HR_CNN and MTTS_CAN data
    test = Test_Methods(videoFileName = video_list[i])
    hr_cnn = test.test_deep(method = "HR_CNN")
    mtts_can = test.test_deep(method = "MTTS_CAN")
    #make df for HR_CNN
    hr_cnn_df = pd.DataFrame([range(7, 61), hr_cnn, np.repeat("HR_CNN", 54)]).T
    hr_cnn_df.columns = ["Time", "BPM", "type"]
    #make df for MTTS_CAN
    mtts_can_df = pd.DataFrame([range(7, 61), mtts_can, np.repeat("MTTS_CAN", 54)]).T
    mtts_can_df.columns = ["Time", "BPM", "type"]
    #concatenate all dataframes
    plot_df = pd.concat([plot_df, hr_cnn_df, mtts_can_df])
    plot_df = plot_df.reset_index(drop=True)

    print("calculating Eulerian...")
    cap2 = cv2.VideoCapture(video_list[i])
    while cap2.isOpened():
        ret, frame = cap2.read()
        if not ret:
            break
        
        #check if video has reached 60 seconds and break
        if cap2.get(cv2.CAP_PROP_POS_MSEC) >= 60000:
            break

        webcam = cap2
        realWidth = 500
        realHeight = 500
        videoWidth = 160
        videoHeight = 120
        videoChannels = 3
        videoFrameRate = 15
        webcam.set(3, realWidth)
        webcam.set(4, realHeight)

        # Color Magnification Parameters
        levels = 3
        alpha = 170
        minFrequency = 1.0
        maxFrequency = 2.0
        bufferSize = 150
        bufferIndex = 0

        # Output Display Parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        loadingTextLocation = (20, 30)
        bpmTextLocation = (videoWidth // 2 + 5, 30)
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 2
        boxColor = (0, 255, 0)
        boxWeight = 3

        # Initialize Gaussian Pyramid
        firstFrame = np.zeros((300, 300, videoChannels))
        firstGauss = buildGauss(firstFrame, levels + 1)[levels]
        # firstGauss = buildLaplacian(firstFrame, levels+1)[levels]
        videoGauss = np.zeros(
            (bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels)
        )
        fourierTransformAvg = np.zeros((bufferSize))

        # Bandpass Filter for Specified Frequencies
        frequencies = (
            (1.0 * videoFrameRate) * np.arange(bufferSize) / (1.0 * bufferSize)
        )
        mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

        # Heart Rate Calculation Variables
        bpmCalculationFrequency = 15
        bpmBufferIndex = 0
        bpmBufferSize = 10
        bpmBuffer = np.zeros((bpmBufferSize))

        network = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
        startY = 0
        endY = 0
        startX = 0
        endX = 0

        FIXED_BOX = False
        fps = cap2.get(cv2.CAP_PROP_FPS)

        bpms=[]
        bpms_hist = []
        bpms_second = [0]
        
        time_idx = 1

        i = 0
        while True:
            ret, frame = webcam.read()
            if ret == False:
                break
            
            (h, w) = frame.shape[:2]
            if not FIXED_BOX:
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(frame, (300, 300)),
                    1.0,
                    (300, 300),
                    (104.0, 177.0, 123.0),
                )
                # Pass blot through network to perform facial detection
                network.setInput(blob)
                detections = network.forward()
                count = 0

                for i in range(0, detections.shape[2]):
                    # Extract confidence assoficated with predictions.
                    confidence = detections[0, 0, i, 2]

                    # Filter based on confidence
                    if confidence < 0.5:
                        continue
                    count += 1

                    # compute BBOX
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Draw box
                    text = "{:.2f}%".format(confidence * 100) + ", Count: " + str(count)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    # cv2.rectangle(frame, (startX, startY),
                    #               (endX, endY), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        text,
                        (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 0),
                        2,
                    )
            else:
                startX = 720
                startY = 150
                endX = 1120
                endY = 500
            detectionFrame = frame[startY:endY, startX:endX, :]

            # Construct Gaussian Pyramid
            pyramid = buildGauss(detectionFrame, levels + 1)[levels]
            # resize pyramid to fit videoGauss
            pyramid = cv2.resize(
                pyramid, (firstGauss.shape[0], firstGauss.shape[1])
            )

            videoGauss[bufferIndex] = pyramid
            fourierTransform = np.fft.fft(videoGauss, axis=0)

            # Bandpass Filter
            fourierTransform[mask == False] = 0

            # Grab a Pulse
            if bufferIndex % bpmCalculationFrequency == 0:
                i = i + 1
                for buf in range(bufferSize):
                    fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
                hz = frequencies[np.argmax(fourierTransformAvg)]
                bpm = 60.0 * hz
                bpmBuffer[bpmBufferIndex] = bpm
                bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

            # Amplify
            filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
            filtered = filtered * alpha

            # Reconstruct Resulting Frame
            filteredFrame = reconstructFrame(
                filtered,
                bufferIndex,
                levels,
                videoHeight=endY - startY,
                videoWidth=endX - startX,
            )
            filteredFrame = cv2.resize(
                filteredFrame, (endX - startX, endY - startY)
            )
            outputFrame = detectionFrame + filteredFrame
            outputFrame = cv2.convertScaleAbs(outputFrame)

            bufferIndex = (bufferIndex + 1) % bufferSize

            #if display_pyramid:
            frame[startY:endY, startX:endX, :] = outputFrame
            
            cv2.rectangle(
                frame, (startX, startY), (endX, endY), boxColor, boxWeight
            )
            if i > bpmBufferSize:
                cv2.putText(
                    frame,
                    "BPM: %d" % bpmBuffer.mean(),
                    bpmTextLocation,
                    font,
                    fontScale,
                    fontColor,
                    lineType,
                )
            else:
                cv2.putText(
                    frame,
                    "Calculating BPM...",
                    loadingTextLocation,
                    font,
                    fontScale,
                    fontColor,
                    lineType,
                )

            frame = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )  # RGB Format to support streamlit
            bpms.append(bpmBuffer.mean())
            bpms_hist.append(bpmBuffer.mean())
            #if len(bpms) > 200:
                #   bpms = bpms[-200:]
            # get hr for each second
            if len(bpms) > 30:
                bpm = np.mean(bpms[-30:])
                bpms_second.append(bpm)
                new_row = pd.DataFrame([[time_idx, bpm, "Eulerian"]], columns=["Time", "BPM", "type"])
                plot_df = pd.concat([plot_df, new_row])
                time_idx += 1
                bpms = []
    plot_df.BPM = plot_df.BPM.astype(float)
    fig = px.line(plot_df[plot_df["Time"].astype(int) > 6], x="Time", y="BPM", 
                                    color="type", labels={"x": "Time", "y": "BPM"},
                                    title="Heart Rate Estimation on Bench Data", width=1200, height=600)
    raw_symbols = SymbolValidator().values
    # Take only the string values which are in this order.
    symbols_names = raw_symbols[::-2]
    markers = cycle(symbols_names)

    # set unique marker style for different countries
    fig.update_traces(mode='lines+markers')
    for d in fig.data:
        d.marker.symbol = next(markers)
        d.marker.size = 10
    fig.show()

    #calculate MAE between methods and ground truth
    mae_hr_cnn = mean_absolute_error(plot_df[plot_df["type"] == "Ground Truth"]["BPM"][:54].astype(float), 
                                 plot_df[plot_df["type"] == "HR_CNN"]["BPM"].astype(float))
    mae_mtts_can = mean_absolute_error(plot_df[plot_df["type"] == "Ground Truth"]["BPM"][:54].astype(float),
                                        plot_df[plot_df["type"] == "MTTS_CAN"]["BPM"].astype(float))
    mae_eulerian = mean_absolute_error(plot_df[plot_df["type"] == "Ground Truth"]["BPM"][:51].astype(float),
                                            plot_df[plot_df["type"] == "Eulerian"]["BPM"][7:58].astype(float))
    print("MAE HR_CNN: ", mae_hr_cnn)
    print("MAE MTTS_CAN: ", mae_mtts_can)
    print("MAE Eulerian: ", mae_eulerian)
    mae_hr_cnn_list.append(mae_hr_cnn)
    mae_mtts_can_list.append(mae_mtts_can)
    mae_eulerian_list.append(mae_eulerian)

#print all MAEs
print("MAE HR_CNN: ", mae_hr_cnn_list)
print("MAE MTTS_CAN: ", mae_mtts_can_list)
print("MAE Eulerian: ", mae_eulerian_list)

print("MEAN MAE HR_CNN: ", np.mean(mae_hr_cnn_list))
print("MEAN MAE MTTS_CAN: ", np.mean(mae_mtts_can_list))
print("MEAN MAE Eulerian: ", np.mean(mae_eulerian_list))


