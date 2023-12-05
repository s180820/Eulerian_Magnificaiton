import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import plotly.express as px  # interactive charts
import tempfile

PROTOTXT_PATH = "../../Models/Facial_recognition/deploy_prototxt.txt"
MODEL_PATH = "../../Models/Facial_recognition/res10_300x300_ssd_iter_140000.caffemodel"

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

def video_upload(video_data=None):
    st.title("Video upload")
    frame_placeholder = st.empty()
    # matplotlib.use('TkAgg')
    placeholder = st.empty()
    #data = st.file_uploader(" ", accept_multiple_files=False)
    if not video_data:
        st.write("Upload a file before continuing")
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_data.read())
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
            ax1 = fig.add_subplot(1, 1, 1)
            fps = cap.get(cv2.CAP_PROP_FPS)

            bpms = []

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
                        # fig = px.line(x=np.arange(len(bpms)), y=bpms, labels={"x": "Time", "y": "BPM"})
                        # st.write(fig)
                        st.image(frame, channels="RGB")
                    with fig_col2:
                        st.markdown("### BPM Statistics")
                        bpm_df = pd.DataFrame(bpms, columns=["BPM"])
                        bpm_df = bpm_df.describe()
                        bpm_df = bpm_df.drop(["count", "min", "25%", "50%", "75%"])
                        bpm_df = bpm_df.rename(
                            index={
                                "mean": "Mean",
                                "max": "Max",
                                "std": "Standard Deviation",
                            }
                        )
                        bpm_df = bpm_df.T
                        bpm_df = bpm_df.round(2)
                        st.write(bpm_df)
                    fig = px.line(
                            x=np.arange(len(bpms)),
                            y=bpms,
                            labels={"x": "Frame", "y": "BPM"},
                        )
                    st.write(fig)

                # st.write(fig)
                # frame_placeholder.image(frame, channels="RGB")
        cap.release()
        cv2.destroyAllWindows()
