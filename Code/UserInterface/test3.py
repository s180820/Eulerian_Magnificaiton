# facial_tracking.py
import cv2
import numpy as np
import dlib
import time
import os
import threading

PROTOTXT_PATH = "../../Models/Facial_recognition/deploy_prototxt.txt"
MODEL_PATH = "../../Models/Facial_recognition/res10_300x300_ssd_iter_140000.caffemodel"


class EulerianMagnification:
    """
    This class implements Eulerian magnification on a video stream.
    """

    def __init__(self, cap) -> None:
        """
        Initialize the class instance of Eulerian magnification.
        """
        self.webcam = cap
        self.realWidth = 500
        self.realHeight = 600
        self.videoWidth = 160
        self.videoHeight = 120
        self.videoChannels = 3
        self.videoFrameRate = 15
        self.webcam.set(3, self.realWidth)
        self.webcam.set(4, self.realHeight)
        self.Frame = None

        self.face_ids = {}
        self.current_face_id = 0
        # Color magnification parameters
        self.levels = 3
        self.alpha = 170
        self.minFrequency = 1.0
        self.maxFrequency = 2.0
        self.bufferSize = 150
        self.bufferIdx = 0

        self.videoGauss, self.firstGauss = self.init_gauss_pyramid()

        # Output display parameters
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.loadingTextLocation = (20, 30)
        self.bpmTextLocation = (self.videoWidth // 2 + 5, 30)
        self.fontScale = 1
        self.fontColor = (255, 255, 255)
        self.lineType = 2
        self.boxColor = (0, 255, 0)
        self.boxWeight = 3

        # Heart Rate Calculation Variables
        self.bpmCalculationFrequency = 15
        self.bpmBufferIndex = 0
        self.bpmBufferSize = 10
        self.bpmBuffer = np.zeros((self.bpmBufferSize))

    def detect_faces(self, network):
        """
        This function takes in a frame and a network and returns a list of faces
        detected in the frame.

        input: frame, network
        output: faces
        """
        (h, w) = self.frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(self.frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        network.setInput(blob)
        detections = network.forward()
        faces = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter based on confidence.
            if confidence > 0.5:
                # Compute boxes if confident of a face and append to face array.
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # Add confidence to the bbox for easy tracking
                faces.append((startX, startY, endX, endY, confidence))
        return faces

    def reconstructFrame(self, pyramid, index, videoWidth=160, videoHeight=120):
        filteredFrame = pyramid[index]
        for level in range(self.levels):
            filteredFrame = cv2.pyrUp(filteredFrame)
        filteredFrame = filteredFrame[:videoHeight, :videoWidth]
        return filteredFrame

    def gaussian_triangle_betch(self, detectionFrame, debugFace=None):
        i = 0
        fourierTransformAvg = np.zeros((self.bufferSize))
        bpmBuffer = self.bpmBuffer
        bpmBufferIdx = self.bpmBufferIndex
        # Init pyramid for each face.
        videoGauss, firstGauss = self.init_gauss_pyramid()
        frequencies, mask = self.initBandPassFilter()
        pyramid = self.buildGauss(detectionFrame)[self.levels]
        # print(pyramid.shape, debugFace[5])
        pyramid = cv2.resize(pyramid, (firstGauss.shape[0], firstGauss.shape[1]))

        videoGauss[self.bufferIdx] = pyramid
        fourierTransform = np.fft.fft(videoGauss, axis=0)

        # Apply bandpass filter
        fourierTransform[mask == False] = 0

        # Grab a pulse (GAP)

        if self.bufferIdx % self.bpmCalculationFrequency == 0:
            i += 1
            for buf in range(self.bufferSize):
                fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
            hz = frequencies[np.argmax(fourierTransformAvg)]
            bpm = 60.0 * hz
            bpmBuffer[bpmBufferIdx] = bpm
            bpmBufferIdx = (bpmBufferIdx + 1) % self.bpmBufferSize
        # Amplify the signal
        filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
        filtered = filtered * self.alpha

        return filtered

    def apply_bbox(self, faces):
        """
        This function takes in a frame and a list of faces and draws bounding boxes
        around the faces.

        input: frame, faces
        output: None
        """
        for startX, startY, endX, endY, confidence in faces:
            cv2.rectangle(self.frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            face_center = ((startX + endX) // 2, (startY + endY) // 2)
            face_id = None

            for id, center in self.face_ids.items():
                distance = (
                    (face_center[0] - center[0]) ** 2
                    + (face_center[1] - center[1]) ** 2
                ) ** 0.5
                if distance <= 90:
                    face_id = id
                    break
            if face_id is None:
                face_id = self.current_face_id
                self.current_face_id += 1
                self.face_ids[face_id] = face_center

            text = f"Confidence: {confidence:.2f}, face_ID {face_id}"
            cv2.putText(
                self.frame,
                text,
                (startX, startY - 10 if startY - 10 > 10 else startY + 10),
                self.font,
                0.45,
                self.boxColor,
                2,
            )
            cv2.imshow("Facial Tracking", self.frame)

            # Frame of detection DEBUG ##############################################
            debugFace = (
                startX,
                startY,
                endX,
                endY,
                confidence,
                faces.index((startX, startY, endX, endY, confidence)),
            )

            detectionFrame = self.frame[startY:endY, startX:endX, :]

    def buildGauss(self, firstframe):
        pyramid = [firstframe]
        for level in range(self.levels + 1):
            frame = cv2.pyrDown(firstframe)
            pyramid.append(frame)
        return pyramid

    def initBandPassFilter(self):
        frequencies = (
            (1.0 * self.videoFrameRate)
            * np.arange(self.bufferSize)
            / (1.0 * self.bufferSize)
        )
        mask = (frequencies >= self.minFrequency) & (frequencies <= self.maxFrequency)
        return frequencies, mask

    def init_gauss_pyramid(self):
        firstFrame = np.zeros((300, 300, self.videoChannels))
        firstGauss = self.buildGauss(firstFrame)[self.levels]
        videoGauss = np.zeros(
            (
                self.bufferSize,
                firstGauss.shape[0],
                firstGauss.shape[1],
                self.videoChannels,
            )
        )
        return videoGauss, firstGauss

    def start_facial_tracking(self):
        """
        This function takes in a video capture object and starts facial tracking
        on the video.

        input: cap
        output: None
        """
        network = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

        current_face_id = 0
        while True:
            ret, self.frame = self.webcam.read()
            if not ret:
                break

            # Facial tracking using openCV
            faces = self.detect_faces(network)
            self.apply_bbox(faces)
            cv2.imshow("Facial Tracking", self.frame)

            # TODO IMPLEMENT HEART RATE CALCULATION

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.exit_event.set()
                break

        self.webcam.release()
        cv2.destroyAllWindows()

        for thread in self.threads.values():
            thread.join()


if __name__ == "__main__":
    video = cv2.VideoCapture(0)
    eulerian = EulerianMagnification(video)
    eulerian.start_facial_tracking()
