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


def bboxBuilder(detections, i, w, h, confidence, count, frame):
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    # Draw box
    text = "{:.2f}%".format(confidence * 100) + ", Count: " + str(count)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(
        frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2
    )


def facial_recognition(frame, network, blob, i, w, h):
    network.setInput(blob)
    detections = network.forward()
    count = 0

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter based on confidence.
        if confidence < 0.5:
            continue
        count += 1
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        # Draw box
        text = "{:.2f}%".format(confidence * 100) + ", Count: " + str(count)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(
            frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2
        )
    detectionFrame = frame[startY:endY, startX:endX, :]


def start_video_feed(cap, display_pyramid=True):
    # Default parameters
    webcam = cap
    realWidth = 320
    realHeight = 240
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

    ## Intializing ##
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

    ## Loop of videos
    i = 0
    while True:
        ret, frame = webcam.read()
        if ret == False:
            break

        (h, w) = frame.shape[:2]
        # Define blob.
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
