"""
Webcam Heart Rate Monitor
Gilad Oved
December 2018
"""

import numpy as np
import cv2
import sys

PROTOTXT_PATH = "../../Models/Facial_recognition/deploy_prototxt.txt"
MODEL_PATH = "../../Models/Facial_recognition/res10_300x300_ssd_iter_140000.caffemodel"


# Helper Methods
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


def start_video_feed(cap, display_pyramid=True):
    # Webcam Parameters
    # webcam = None
    # if len(sys.argv) == 2:
    #   webcam = cv2.VideoCapture(sys.argv[1])
    # else:
    #   webcam = cv2.VideoCapture(0)
    webcam = cap
    realWidth = 320
    realHeight = 240
    videoWidth = 160
    videoHeight = 120
    videoChannels = 3
    videoFrameRate = 15
    webcam.set(3, realWidth)
    webcam.set(4, realHeight)

    # Output Videos
    if len(sys.argv) != 2:
        originalVideoFilename = "original.mov"
        originalVideoWriter = cv2.VideoWriter()
        originalVideoWriter.open(
            originalVideoFilename,
            cv2.VideoWriter_fourcc("j", "p", "e", "g"),
            videoFrameRate,
            (realWidth, realHeight),
            True,
        )

    outputVideoFilename = "output.mov"
    outputVideoWriter = cv2.VideoWriter()
    outputVideoWriter.open(
        outputVideoFilename,
        cv2.VideoWriter_fourcc("j", "p", "e", "g"),
        videoFrameRate,
        (realWidth, realHeight),
        True,
    )

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
    frequencies = (1.0 * videoFrameRate) * np.arange(bufferSize) / (1.0 * bufferSize)
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

    i = 0
    while True:
        ret, frame = webcam.read()
        if ret == False:
            break

        if len(sys.argv) != 2:
            originalFrame = frame.copy()
            originalVideoWriter.write(originalFrame)

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
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
                frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2
            )
        detectionFrame = frame[startY:endY, startX:endX, :]
        """
        Not an Option to update videoGauss, how do I make Video gauss able to take in different size frames?
        """
        # secondFrame = np.zeros((endY-startY, endX-startX, videoChannels))
        # secondGauss = buildGauss(secondFrame, levels+1)[levels]
        # videoGauss = np.zeros((bufferSize, secondGauss.shape[0], secondGauss.shape[1], videoChannels))

        # Construct Gaussian Pyramid
        pyramid = buildGauss(detectionFrame, levels + 1)[levels]
        print(firstGauss.shape)
        # resize pyramid to fit videoGauss
        pyramid = cv2.resize(pyramid, (firstGauss.shape[0], firstGauss.shape[1]))

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
        filteredFrame = cv2.resize(filteredFrame, (endX - startX, endY - startY))
        outputFrame = detectionFrame + filteredFrame
        outputFrame = cv2.convertScaleAbs(outputFrame)

        bufferIndex = (bufferIndex + 1) % bufferSize

        frame[startY:endY, startX:endX, :] = outputFrame
        if display_pyramid:
            cv2.rectangle(frame, (startX, startY), (endX, endY), boxColor, boxWeight)
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

        outputVideoWriter.write(frame)

        if len(sys.argv) != 2:
            cv2.imshow("Webcam Heart Rate Monitor", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    webcam.release()
    cv2.destroyAllWindows()
    outputVideoWriter.release()
    if len(sys.argv) != 2:
        originalVideoWriter.release()


if __name__ == "__main__":
    video = cv2.VideoCapture(0)
    # video = cv2.VideoCapture("c920_00_02.avi")
    start_video_feed(video)
