"""
    Authors:
        - Frederik Peetz-Schou Larsen
        - Gustav Gamst Larsen

    Year: 
        - 2023
    
    Description: 
        - As part of "Advanced Project in Digital Media Engineering" on DTU 
"""

import cv2
import streamlit as st
import dlib
import numpy as np


class MultifaceEulerianMagnification:
    """
    Class to perform concurrent eulerian magnification
    on multiple faces in a video stream
    Attributes:
        detector: dlib face detector
        predictor: dlib facial landmarks predictor
        facerec: dlib face recognition model
        face_encodings: dictionary to store face encodings
        labels: dictionary to store face labels
        cap: webcam object
        frame_counter: counter to keep track of frames
        realWidth: width of the webcam frame
        realHeight: height of the webcam frame
        videoWidth: width of the video frame
        videoHeight: height of the video frame
        videoChannels: number of channels in the video frame
        videoFrameRate: frame rate of the video
        frame: current frame
        levels: number of levels in the gaussian pyramid
        alpha: alpha value for color magnification
        minFrequency: minimum frequency for color magnification
        maxFrequency: maximum frequency for color magnification
        bufferSize: size of the buffer
        bufferIdx: dictionary to store buffer indices
        font: font for the text
        loadingTextLocation: location of the loading text
        bpmTextLocation: location of the bpm text
        fontScale: font scale
        fontColor: font color
        lineType: line type
        boxColor: color of the bounding box
        boxWeight: weight of the bounding box
        bpmCalculationFrequency: frequency of bpm calculation
        bpmBufferIndex: dictionary to store bpm buffer indices
        bpmBufferSize: size of the bpm buffer
        bpmBuffer: dictionary to store bpm buffers
        BPMs: dictionary to store BPMs
        firstFrame: first frame of the video
        firstGauss: first gaussian pyramid
        videoGauss: dictionary to store gaussian pyramids
        fourierTransformAvg: dictionary to store fourier transforms
        frequencies: frequencies for color magnification
        mask: mask for color magnification
        i: counter to keep track of frames

    Methods:
        buildGauss: builds a gaussian pyramid
        unpack_coordinates: unpacks the bounding box coordinates
        estimate_heart_rate: grabs the pulse from the fourier transform
        eulerian_magnification: performs eulerian magnification
        runner: runs the eulerian magnification
        recognize_faces: recognizes faces in the video stream


    """

    def __init__(self):
        # Load the pre-trained face detection model from dlib
        self.detector = dlib.get_frontal_face_detector()

        # Load the facial landmarks predictor
        self.predictor_path = (
            "../../Models/Facial_recognition/shape_predictor_68_face_landmarks.dat"
        )
        self.predictor = dlib.shape_predictor(self.predictor_path)

        # Load the pre-trained face recognition model
        self.face_rec_model_path = (
            "../../Models/Facial_recognition/dlib_face_recognition_resnet_model_v1.dat"
        )
        self.facerec = dlib.face_recognition_model_v1(self.face_rec_model_path)

        # Create a dictionary to store face encodings and labels
        self.face_encodings = {}
        self.labels = {}

        # Open the webcam
        self.cap = cv2.VideoCapture(0)

        # Variables
        self.frame_counter = 0

        # Video variables
        self.realWidth = 500
        self.realHeight = 600
        self.videoWidth = 160
        self.videoHeight = 120
        self.videoChannels = 3
        self.videoFrameRate = 30
        self.cap.set(3, self.realWidth)
        self.cap.set(4, self.realHeight)
        self.frame = None

        # Color magnification parameters
        self.levels = 3
        self.alpha = 170
        self.minFrequency = 1.0
        self.maxFrequency = 2.0
        self.bufferSize = 150
        self.bufferIdx = {}

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
        self.bpmBufferIndex = {}
        self.bpmBufferSize = 10
        self.bpmBuffer = {}

        self.BPMs = {}

        self.firstFrame = np.zeros((50, 50, self.videoChannels))
        self.firstGauss = self.buildGauss(self.firstFrame)[self.levels]

        self.videoGauss = {}

        self.fourierTransformAvg = {}

        self.frequencies = (
            (1.0 * self.videoFrameRate)
            * np.arange(self.bufferSize)
            / (1.0 * self.bufferSize)
        )
        self.mask = (self.frequencies >= self.minFrequency) & (
            self.frequencies <= self.maxFrequency
        )
        self.i = 0

    def get_bpm_over_time(self):
        """
        Used for StreamLit statistics
        """
        return self.bpmBuffer

    def buildGauss(self, frame):
        pyramid = [frame]
        for level in range(self.levels + 1):
            frame = cv2.pyrDown(frame)
            pyramid.append(frame)
        return pyramid

    def unpack_coordinates(self, bbox):
        start_point, end_point = bbox
        startX, startY = start_point
        endX, endY = end_point
        return startY, endY, startX, endX

    def process_frame_streamlit(self, frame):
        self.frame = frame
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        bounding_boxes = []

        for face in faces:
            shape = self.predictor(gray, face)

            # Get face encofing
            face_encoding = self.facerec.compute_face_descriptor(self.frame, shape)

            # check if the face matches any known face
            match = None
            for label, encoding in self.face_encodings.items():
                if np.linalg.norm(np.array(encoding) - np.array(face_encoding)) < 0.6:
                    match = label
                    break
            # If the face is not recognized
            if match is None:
                label = len(self.face_encodings) + 1
                self.face_encodings[label] = face_encoding
                self.labels[label] = f"Person {label}"
            # Draw rect
            bbox = ((face.left(), face.top()), (face.right(), face.bottom()))
            bounding_boxes.append((bbox, match))

            cv2.rectangle(frame, bbox[0], bbox[1], (0, 255, 0), 2)
            # Display the label
            if match is not None:
                label_text = self.labels[match]
                bpm_text = f"BPM: {self.BPMs.get(match, [0])[-1]:.2f}"  # Display a static BPM value
                cv2.putText(
                    self.frame,
                    label_text,
                    (bbox[0][0], bbox[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    self.frame,
                    bpm_text,
                    (
                        bbox[0][0],
                        bbox[1][1] + 20,
                    ),  # Adjust the vertical position
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,  # Adjust the font scale
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    self.frame,
                    "Unknown",
                    (bbox[0][0], bbox[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
            # # Process bboxes
            for bbox, face_id in bounding_boxes:
                self.runner(bbox=bbox, face_id=face_id)
            # print(self.BPMs)
        return self.frame

    def estimate_heart_rate(self, fourierTransform, face_id):
        if face_id not in self.bpmBufferIndex:
            print(f"[DEBUG - Pulse] {face_id} not found in bpmBufferIndex. Setting 0")
            self.bpmBufferIndex[face_id] = 0
        if self.bufferIdx[face_id] % self.bpmCalculationFrequency == 0:
            self.i += 1
            for buf in range(self.bufferSize):
                self.fourierTransformAvg[face_id][buf] = np.real(
                    fourierTransform[buf].mean()
                )
            hz = self.frequencies[np.argmax(self.fourierTransformAvg[face_id])]
            if hz > 0:
                bpm = 60 * hz
                self.bpmBuffer[face_id][self.bpmBufferIndex[face_id]] = bpm
                self.bpmBufferIndex[face_id] = (
                    self.bpmBufferIndex[face_id] + 1
                ) % self.bpmBufferSize
                return bpm

    def eulerian_magnification(
        self, detectionframe, startY, endY, startX, endX, face_id
    ):
        # print(f"[DEBUG - Eulerian] Performing eulerian Magnification for {face_id}")
        pyramid = self.buildGauss(detectionframe)[self.levels]
        pyramid = cv2.resize(
            pyramid, (self.firstGauss.shape[0], self.firstGauss.shape[1])
        )

        # Initialize fouierTransform avg, and videos to keep for each {face_id}
        if face_id not in self.bufferIdx:
            print(f"[DEBUG - Eulerian] {face_id} not found in bufferIdx. Setting 0")
            self.bufferIdx[face_id] = 0
            self.videoGauss[face_id] = np.zeros(
                (
                    self.bufferSize,
                    self.firstGauss.shape[0],
                    self.firstGauss.shape[1],
                    self.videoChannels,
                )
            )
            self.fourierTransformAvg[face_id] = np.zeros((self.bufferSize))
            self.bpmBuffer[face_id] = np.zeros((self.bpmBufferSize))
            self.BPMs[face_id] = []

        # If already exist or just created -- Continue computing fft.
        self.videoGauss[face_id][self.bufferIdx[face_id]] = pyramid

        fourierTransform = np.fft.fft(self.videoGauss[face_id], axis=0)
        fourierTransform[self.mask == False] = 0

        bpm = self.estimate_heart_rate(
            fourierTransform=fourierTransform, face_id=face_id
        )
        # print(f"[VERBOSE] Heartrate of person {face_id}: {bpm}")
        if face_id not in self.BPMs:
            # initialize at an empty list
            self.BPMs[face_id] = []
        if bpm is not None:
            self.BPMs[face_id].append(bpm)

        self.bufferIdx[face_id] = (self.bufferIdx[face_id] + 1) % self.bufferSize

    def runner(self, bbox, face_id):
        startY, endY, startX, endX = self.unpack_coordinates(bbox)

        detectionFrame = self.frame[startY:endY, startX:endX, :]

        self.eulerian_magnification(
            detectionframe=detectionFrame,
            startY=startY,
            startX=startX,
            endY=endY,
            endX=endX,
            face_id=face_id,
        )

    def recognize_faces(self):
        while True:
            ret, self.frame = self.cap.read()
            self.frame_counter += 1

            if self.frame_counter % 1 == 0:
                # Convert the frame to grayscale for face detection
                gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

                # Detect faces in the frame
                faces = self.detector(gray)

                # List to store bounding boxes and face IDs
                bounding_boxes = []

                for face in faces:
                    # Get facial landmarks
                    shape = self.predictor(gray, face)

                    # Get the face encoding
                    face_encoding = self.facerec.compute_face_descriptor(
                        self.frame, shape
                    )

                    # Check if the face matches any known face
                    match = None
                    for label, encoding in self.face_encodings.items():
                        if (
                            np.linalg.norm(np.array(encoding) - np.array(face_encoding))
                            < 0.6
                        ):  # You can adjust this threshold
                            match = label
                            break

                    # If the face is not recognized, assign a new label
                    if match is None:
                        label = len(self.face_encodings) + 1
                        self.face_encodings[label] = face_encoding
                        self.labels[label] = f"Person {label}"

                    # Draw a rectangle around the face
                    bbox = ((face.left(), face.top()), (face.right(), face.bottom()))
                    bounding_boxes.append((bbox, match))

                    cv2.rectangle(
                        self.frame,
                        bbox[0],
                        bbox[1],
                        (0, 255, 0),
                        2,
                    )

                    # Display the label
                    if match is not None:
                        label_text = self.labels[match]
                        bpm_text = f"BPM: {self.BPMs.get(match, [0])[-1]:.2f}"  # Display a static BPM value
                        cv2.putText(
                            self.frame,
                            label_text,
                            (bbox[0][0], bbox[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            self.frame,
                            bpm_text,
                            (
                                bbox[0][0],
                                bbox[1][1] + 20,
                            ),  # Adjust the vertical position
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,  # Adjust the font scale
                            (0, 255, 0),
                            2,
                        )

                    else:
                        cv2.putText(
                            self.frame,
                            "Unknown",
                            (bbox[0][0], bbox[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2,
                        )

                # Process bounding boxes and face IDs as needed
                for bbox, face_id in bounding_boxes:
                    self.runner(bbox=bbox, face_id=face_id)
                # Display the frame
                cv2.imshow("Webcam", self.frame)

                # Break the loop when 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        # Release the webcam and close all windows
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Instantiate the class and run the facial recognition
    eulerian = MultifaceEulerianMagnification()
    eulerian.recognize_faces()  # Running method -- Will call concurrent methods to perform Eulerian Magnification
