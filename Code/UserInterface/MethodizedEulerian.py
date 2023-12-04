import numpy as np
import cv2
import sys
import streamlit as st

PROTOTXT_PATH = "../../Models/Facial_recognition/deploy_prototxt.txt"
MODEL_PATH = "../../Models/Facial_recognition/res10_300x300_ssd_iter_140000.caffemodel"


# Helper Methods
class EulerianMagnification:
    """
    This class implements Eulerian magnification on a video stream.
    """

    def __init__(self, cap=None) -> None:
        """
        Initialize the class instance of Eulerian magnification.
        """
        self.webcam = cap
        self.realWidth = 1920
        self.realHeight = 1080
        self.videoWidth = 160
        self.videoHeight = 120
        self.videoChannels = 3
        self.videoFrameRate = 30
        # self.webcam.set(3, self.realWidth)
        # self.webcam.set(4, self.realHeight)
        self.Frame = None
        self.detectionFrames = {}
        self.display_pyramid = True

        self.face_ids = {}
        self.current_face_id = 0
        # Color magnification parameters
        self.levels = 3
        self.alpha = 170
        self.minFrequency = 1.0
        self.maxFrequency = 2.0
        self.bufferSize = 150
        self.bufferIdx = 0

        self.count = 0

        # self.videoGauss, self.firstGauss = self.init_gauss_pyramid()

        # Output display parameters
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.loadingTextLocation = (20, 30)
        self.bpmTextLocation = (self.videoWidth // 2 + 5, 30)
        self.fontScale = 1
        self.lineType = 2
        self.boxColor = (0, 255, 0)
        self.boxWeight = 3

        # Heart Rate Calculation Variables
        self.bpmCalculationFrequency = 15
        self.bpmBufferIndex = 0
        self.bpmBufferSize = 10
        self.bpmBuffer = np.zeros((self.bpmBufferSize))

        self.firstFrame = np.zeros((50, 50, self.videoChannels))
        self.fourierTransformAvg = np.zeros((self.bufferSize))

        self.frequencies = (
            (1.0 * self.videoFrameRate)
            * np.arange(self.bufferSize)
            / (1.0 * self.bufferSize)
        )
        self.mask = (self.frequencies >= self.minFrequency) & (
            self.frequencies <= self.maxFrequency
        )

        # Heart rate calc variables.
        self.bpmCalculationFrequency = 15
        self.bpmBufferIndex = 0
        self.bpmBufferSize = 10
        self.bpmBuffer = np.zeros((self.bpmBufferSize))

        self.i = 0
        self.network = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

        # Init EulerianMagnification
        self.firstFrame = np.zeros((300, 300, self.videoChannels))
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

    def compute_bbox(self, detections, h, w, i):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        return (startX, startY, endX, endY)

    def image_recog(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        self.network.setInput(blob)
        detections = self.network.forward()
        self.count = 0

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence < 0.5:
                continue
            self.count += 1
            # compute bbox
            (startX, startY, endX, endY) = self.compute_bbox(detections, h, w, i)

            return startX, startY, endX, endY, confidence

    def apply_bbox(self, box, frame, confidence, BPM):
        startX, startY, endX, endY = box
        if self.i > self.bpmBufferSize:
            if BPM > 190 or BPM > 40 and confidence * 100 > 50:
                text = "Confidence: {:.2f}%".format(
                    confidence * 100
                ) + "   BPM: {}".format(int(BPM))
        else:
            text = (
                "Confidence: {:.2f}%".format(confidence * 100)
                + "   BPM: Calculating..."
            )

        # Center the text above the bounding box
        text_size = cv2.getTextSize(text, self.font, self.fontScale, self.lineType)[0]
        text_x = startX + (endX - startX - text_size[0]) // 2
        text_y = startY - 10 if startY - 10 > 10 else startY + 10

        cv2.rectangle(
            frame, (startX, startY), (endX, endY), self.boxColor, self.boxWeight
        )
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.fontScale,
            (0, 255, 0),
            2,
        )

    def buildGauss(self, frame):
        pyramid = [frame]
        for level in range(self.levels + 1):
            frame = cv2.pyrDown(frame)
            pyramid.append(frame)
        return pyramid

    def grab_pulse(self, fourierTransform):
        if self.bufferIdx % self.bpmCalculationFrequency == 0:
            self.i += 1
            for buf in range(self.bufferSize):
                self.fourierTransformAvg[buf] = np.real(fourierTransform[buf].mean())
            hz = self.frequencies[np.argmax(self.fourierTransformAvg)]
            bpm = 60 / hz
            self.bpmBuffer[self.bpmBufferIndex] = bpm
            self.bpmBufferIndex = (self.bpmBufferIndex + 1) % self.bpmBufferSize

            return bpm

    def reconstructFrame(self, pyramid, index, videoWidth=160, videoHeight=120):
        filteredFrame = pyramid[index]
        for level in range(self.levels):
            filteredFrame = cv2.pyrUp(filteredFrame)
        filteredFrame = filteredFrame[:videoHeight, :videoWidth]
        return filteredFrame

    def eulerianMagnification(self, detectionFrame, startY, endY, startX, endX):
        pyramid = self.buildGauss(detectionFrame)[self.levels]
        pyramid = cv2.resize(
            pyramid, (self.firstGauss.shape[0], self.firstGauss.shape[1])
        )
        self.videoGauss[self.bufferIdx] = pyramid
        fourierTranform = np.fft.fft(self.videoGauss, axis=0)
        fourierTranform[self.mask == False] = 0

        # Get bpm
        bpm = self.grab_pulse(fourierTranform)

        filtered = np.real(np.fft.ifft(fourierTranform, axis=0))
        filtered = filtered * self.alpha

        filteredFrame = self.reconstructFrame(
            filtered,
            self.bufferIdx,
            videoHeight=endY - startY,
            videoWidth=endX - startX,
        )
        filteredFrame = cv2.resize(filteredFrame, (endX - startX, endY - startY))

        outputFrame = detectionFrame + filteredFrame
        outputFrame = cv2.convertScaleAbs(outputFrame)
        self.bufferIdx = (self.bufferIdx + 1) % self.bufferSize
        return bpm, outputFrame

    def start_video_feed(self):
        """Start the video feed and calculate heart rate."""

        while True:
            ret, frame = self.webcam.read()

            if ret == False:
                break

            # Get bbox of face
            startX, startY, endX, endY, confidence = self.image_recog(frame)
            # Apply
            # print(confidence)
            self.apply_bbox(
                (startX, startY, endX, endY),
                frame,
                confidence,
                BPM=self.bpmBuffer.mean(),
            )

            # Detection frame
            detectionFrame = frame[startY:endY, startX:endX, :]
            bpm, outputframe = self.eulerianMagnification(
                detectionFrame, startY, endY, startX, endX
            )

            if len(sys.argv) != 2:
                cv2.imshow("Traditional Eulerian Magnification", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    def get_bpm_over_time(self):
        """
        Used for StremLit statistics
        """
        return self.bpmBuffer

    def start_video_feed_streamlit(self):
        """Start the video feed and calculate heart rate for Streamlit."""

        st.button("Pyramid Off/On dev")  # Placeholder for Pyramid Off/On button

        video_placeholder = st.empty()

        video_started = False

        while True:
            # Check if the Start button is pressed
            if st.button("Start dev") and not video_started:
                video_started = True
                st.write("Video stream started!")

                while True:
                    ret, frame = self.webcam.read()

                    if not ret:
                        st.write("The video capture has ended.")
                        break

                    processed_frame = self.process_frame(frame)

                    # Display the processed frame
                    video_placeholder.image(
                        processed_frame, channels="RGB", use_column_width=True
                    )

                    # Check if the Stop button is pressed
                    if st.button("Stop dev"):
                        video_started = False
                        st.write("Stopping the video feed.")
                        break

            # Wait for a short duration to avoid high CPU usage
            st.experimental_rerun()

    def process_frame(self, frame):
        detection_result = self.image_recog(frame)

        if detection_result is not None:
            startX, startY, endX, endY, confidence = detection_result
            self.apply_bbox(
                (startX, startY, endX, endY),
                frame,
                confidence,
                BPM=self.bpmBuffer.mean(),
            )

            detection_frame = frame[startY:endY, startX:endX, :]
            bpm, output_frame = self.eulerianMagnification(
                detection_frame, startY, endY, startX, endX
            )

            # Add colorizing effect to the original frame
            if self.display_pyramid:
                frame[startY:endY, startX:endX, :] = output_frame

            return frame  # Display the frame with the color filter

        # If face detection is not successful, return the original frame
        return frame


if __name__ == "__main__":
    video = cv2.VideoCapture(0)
    mag = EulerianMagnification(video)
    mag.start_video_feed()
