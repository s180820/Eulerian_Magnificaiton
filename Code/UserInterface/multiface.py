import cv2
import dlib
import numpy as np


class FacialRecognition:
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

        ## OLD
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

        self.firstFrame = np.zeros((50, 50, self.videoChannels))
        self.firstGauss = self.buildGauss(self.firstFrame)[self.levels]

        self.videoGauss = np.zeros(
            (
                self.bufferSize,
                self.firstGauss.shape[0],
                self.firstGauss.shape[1],
                self.videoChannels,
            )
        )

        self.fourierTransformAvg = np.zeros((self.bufferSize))

        self.frequencies = (
            (1.0 * self.videoFrameRate)
            * np.arange(self.bufferSize)
            / (1.0 * self.bufferSize)
        )
        self.mask = (self.frequencies >= self.minFrequency) & (
            self.frequencies <= self.maxFrequency
        )

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

    def eulerian_magnification(
        self, detectionframe, startY, endY, startX, endX, face_id
    ):
        print(f"[DEBUG] Performing eulerian Magnification for {face_id}")
        pyramid = self.buildGauss(detectionframe)[self.levels]
        pyramid = cv2.resize(
            pyramid, (self.firstGauss.shape[0], self.firstGauss.shape[1])
        )

    def test_function_for_person(self, bbox, face_id):
        # Add your custom test function logic here
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

        # print(f"Face ID: {face_id}, Bounding Box: {bbox}")

    def recognize_faces(self):
        while True:
            ret, self.frame = self.cap.read()
            self.frame_counter += 1

            if self.frame_counter % 3 == 0:
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
                        cv2.putText(
                            self.frame,
                            self.labels[match],
                            (bbox[0][0], bbox[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2,
                        )

                        # Call the test function for the recognized person

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
                    self.test_function_for_person(bbox=bbox, face_id=face_id)

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
    facial_recognition = FacialRecognition()
    facial_recognition.recognize_faces()
