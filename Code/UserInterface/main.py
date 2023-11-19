import streamlit as st
import pandas as pd
import numpy as np
import cv2
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px  # interactive charts
import tempfile

st.set_page_config(page_title="Eulerian Magnification", page_icon=":eyeglasses:", layout="wide")
tab1, tab2, tab3, tab4 = st.tabs(["Pre-recorded video", "Live feed", "Validation test", "About"])

PROTOTXT_PATH = "../../Models/Facial_recognition/deploy_prototxt.txt"
MODEL_PATH = "../../Models/Facial_recognition/res10_300x300_ssd_iter_140000.caffemodel"


## Helper functions ""
def markdownreader(file):
    with open("Text/" + file) as f:
        lines = f.readlines()
        for line in lines:
            st.markdown(line, unsafe_allow_html=True)


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


def buildLaplacian(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    for level in range(levels, 0, -1):
        expanded = cv2.pyrUp(pyramid[level])
        laplacian = cv2.subtract(pyramid[level - 1], expanded)
        pyramid[level - 1] = laplacian
    return pyramid


## TABS ##
with tab1:
    markdownreader("Main.md")
    # Data uploade
    frame_placeholder = st.empty()
    #matplotlib.use('TkAgg')
    placeholder = st.empty()
    data = st.file_uploader(" ", accept_multiple_files=False)
    if not data:
        st.write("Upload a file before continuing")
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(data.read())
        cap = cv2.VideoCapture(tfile.name)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("The video capture has ended")
                break
            
            display_pyramid = True
            webcam = cap
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

            fig = plt.figure()
            ax1 = fig.add_subplot(1,1,1)
            fps = cap.get(cv2.CAP_PROP_FPS)

            bpms=[]

            i = 0
            while True:
                ret, frame = webcam.read()
                if ret == False:
                    break

                (h, w) = frame.shape[:2]
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
                detectionFrame = frame[startY:endY, startX:endX, :]
                # """
                # Not an Option to update videoGauss, how do I make Video gauss able to take in different size frames?
                # """
                # secondFrame = np.zeros((endY-startY, endX-startX, videoChannels))
                # secondGauss = buildGauss(secondFrame, levels+1)[levels]
                # videoGauss = np.zeros((bufferSize, secondGauss.shape[0], secondGauss.shape[1], videoChannels))

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

                if display_pyramid:
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
                if len(bpms) > 200:
                    bpms = bpms[-200:]
                # with plt.ion():
                #     ax1.clear()
                #     ax1.plot(bpms)
                #     ax1.set_xlabel('Time')
                #     ax1.set_ylabel('BPM')
                #     plt.pause(0.0001)
                #     with fig_col1:
                #         st.write(fig)
                with placeholder.container():
                    fig_col1, fig_col2 = st.columns(2)
                    with fig_col1:
                        st.markdown("### BPM over time")
                        #fig = px.line(x=np.arange(len(bpms)), y=bpms, labels={"x": "Time", "y": "BPM"})
                        #st.write(fig)
                        st.image(frame, channels="RGB")
                    with fig_col2:
                        st.markdown("### BPM Statistics")
                        bpm_df = pd.DataFrame(bpms, columns=["BPM"])
                        bpm_df = bpm_df.describe()
                        bpm_df = bpm_df.drop(["count", "min", "25%", "50%", "75%"])
                        bpm_df = bpm_df.rename(index={"mean": "Mean", "max": "Max", "std": "Standard Deviation"})
                        bpm_df = bpm_df.T
                        bpm_df = bpm_df.round(2)
                        st.write(bpm_df)
                        fig = px.line(x=np.arange(len(bpms)), y=bpms, labels={"x": "Frame", "y": "BPM"})
                        st.write(fig)
                        
                    
                #st.write(fig)
                #frame_placeholder.image(frame, channels="RGB")
        cap.release()
        cv2.destroyAllWindows()


with tab2:
    # Init
    markdownreader("Webcam.md")
    start_button_pressed = st.button("Start")

    stop_bottom_pressed = st.button("Stop")

    frame_placeholder = st.empty()
    #matplotlib.use('TkAgg')
    placeholder = st.empty()

    if start_button_pressed:
        pyramid_button = st.button("Pyramid Off/On")
        # Test
        if pyramid_button:
            display_pyramid = True
        if not pyramid_button:
            display_pyramid = False
        cap = cv2.VideoCapture(0)
        while cap.isOpened() and not stop_bottom_pressed:
            ret, frame = cap.read()
            if not ret:
                st.write("The video capture has ended")
                break

            webcam = cap
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

            fig = plt.figure()
            ax1 = fig.add_subplot(1,1,1)
            fps = cap.get(cv2.CAP_PROP_FPS)

            bpms=[]

            i = 0
            while True:
                ret, frame = webcam.read()
                if ret == False:
                    break

                (h, w) = frame.shape[:2]
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
                detectionFrame = frame[startY:endY, startX:endX, :]
                # """
                # Not an Option to update videoGauss, how do I make Video gauss able to take in different size frames?
                # """
                # secondFrame = np.zeros((endY-startY, endX-startX, videoChannels))
                # secondGauss = buildGauss(secondFrame, levels+1)[levels]
                # videoGauss = np.zeros((bufferSize, secondGauss.shape[0], secondGauss.shape[1], videoChannels))

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

                if display_pyramid:
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
                if len(bpms) > 200:
                    bpms = bpms[-200:]
                # with plt.ion():
                #     ax1.clear()
                #     ax1.plot(bpms)
                #     ax1.set_xlabel('Time')
                #     ax1.set_ylabel('BPM')
                #     plt.pause(0.0001)
                #     with fig_col1:
                #         st.write(fig)
                with placeholder.container():
                    fig_col1, fig_col2 = st.columns(2)
                    with fig_col1:
                        st.markdown("### BPM over time")
                        #fig = px.line(x=np.arange(len(bpms)), y=bpms, labels={"x": "Time", "y": "BPM"})
                        #st.write(fig)
                        st.image(frame, channels="RGB")
                    with fig_col2:
                        st.markdown("### BPM Statistics")
                        bpm_df = pd.DataFrame(bpms, columns=["BPM"])
                        bpm_df = bpm_df.describe()
                        bpm_df = bpm_df.drop(["count", "min", "25%", "50%", "75%"])
                        bpm_df = bpm_df.rename(index={"mean": "Mean", "max": "Max", "std": "Standard Deviation"})
                        bpm_df = bpm_df.T
                        bpm_df = bpm_df.round(2)
                        st.write(bpm_df)
                        #st.markdown("### BPM over time")
                        fig = px.line(x=np.arange(len(bpms)), y=bpms, labels={"x": "Time", "y": "BPM"})
                        st.write(fig)
                        
                    
                #st.write(fig)
                #frame_placeholder.image(frame, channels="RGB")
        cap.release()
        cv2.destroyAllWindows()
    st.markdown("### Detailed Data View")

with tab3:
    # Init
    markdownreader("Webcam.md")
    start_button_pressed_2 = st.button("Start Test")

    stop_bottom_pressed_2 = st.button("Stop Test")

    frame_placeholder2 = st.empty()
    #matplotlib.use('TkAgg')
    placeholder2 = st.empty()
    # Test
    cap2 = cv2.VideoCapture("Validation/Validation_film.mp4")
    if start_button_pressed_2:
        while cap2.isOpened() and not stop_bottom_pressed_2:
            ret, frame = cap2.read()
            if not ret:
                st.write("The video capture has ended")
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

            fig = plt.figure()
            ax1 = fig.add_subplot(1,1,1)
            fps = cap2.get(cv2.CAP_PROP_FPS)

            bpms=[]
            bpms_hist = []
            bpms_second = [0]

            #calculate ground truth bpm
            rr = pd.read_csv('Validation/P1LC1_Mobi_RR-intervals.rr', header=None)
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
            time_idx = 1

            i = 0
            while True:
                ret, frame = webcam.read()
                if ret == False:
                    break

                (h, w) = frame.shape[:2]
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
                detectionFrame = frame[startY:endY, startX:endX, :]
                # """
                # Not an Option to update videoGauss, how do I make Video gauss able to take in different size frames?
                # """
                # secondFrame = np.zeros((endY-startY, endX-startX, videoChannels))
                # secondGauss = buildGauss(secondFrame, levels+1)[levels]
                # videoGauss = np.zeros((bufferSize, secondGauss.shape[0], secondGauss.shape[1], videoChannels))

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
                # with plt.ion():
                #     ax1.clear()
                #     ax1.plot(bpms)
                #     ax1.set_xlabel('Time')
                #     ax1.set_ylabel('BPM')
                #     plt.pause(0.0001)
                #     with fig_col1:
                #         st.write(fig)
                with placeholder2.container():
                    fig_col1, fig_col2 = st.columns(2)
                    with fig_col1:
                        st.markdown("### BPM over time")
                        #fig = px.line(x=np.arange(len(bpms_second)), y=bpms_second, labels={"x": "Time", "y": "BPM"})
                        #fig = px.line(plot_df, x="Time", y="BPM", color="type", labels={"x": "Time", "y": "BPM"})
                        #st.write(fig)
                        st.image(frame, channels="RGB")
                    with fig_col2:
                        st.markdown("### BPM Statistics")
                        bpm_df = pd.DataFrame(bpms_hist, columns=["BPM"])
                        bpm_df = bpm_df.describe()
                        bpm_df = bpm_df.drop(["count", "min", "25%", "50%", "75%"])
                        bpm_df = bpm_df.rename(index={"mean": "Mean", "max": "Max", "std": "Standard Deviation"})
                        bpm_df = bpm_df.T
                        bpm_df = bpm_df.round(2)
                        st.write(bpm_df)
                        fig = px.line(plot_df, x="Time", y="BPM", color="type", labels={"x": "Time", "y": "BPM"})
                        st.write(fig)
                        
                    
                #st.write(fig)
                #frame_placeholder2.image(frame, channels="RGB")
        cap2.release()
        cv2.destroyAllWindows()
    st.markdown("### Detailed Data View")




with tab4:
    markdownreader("Background.md")
