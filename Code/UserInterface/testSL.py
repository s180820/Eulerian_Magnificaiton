import streamlit as st
import cv2
from MethodizedEulerian import (
    EulerianMagnification,
)  # Assuming you have a class for Eulerian Magnification


class VideoStream:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.eulerian = EulerianMagnification(cap=self.cap)
        self.display_pyramid = True

    def start_video_feed(self):
        while True:
            ret, frame = self.cap.read()

            if not ret:
                st.write("The video capture has ended")
                break

            # Process the frame using your existing logic
            startX, startY, endX, endY, confidence = self.eulerian.image_recog(frame)
            self.eulerian.apply_bbox(
                (startX, startY, endX, endY),
                frame,
                confidence,
                BPM=self.eulerian.bpmBuffer.mean(),
            )

            detectionFrame = frame[startY:endY, startX:endX, :]
            bpm, outputframe = self.eulerian.eulerianMagnification(
                detectionFrame, startY, endY, startX, endX
            )

            if self.display_pyramid:
                cv2.rectangle(
                    frame,
                    (startX, startY),
                    (endX, endY),
                    self.eulerian.boxColor,
                    self.eulerian.boxWeight,
                )

            st.image(frame, channels="BGR", use_column_width=True)

            # Event handling
            if st.button("Pyramid Off/On"):
                self.display_pyramid = not self.display_pyramid

    def stop_video_feed(self):
        self.cap.release()


# Streamlit app
st.title("Video Stream with Eulerian Magnification")

video_stream = VideoStream()

with st.sidebar:
    start_button = st.button("Start dev")
    stop_button = st.button("Stop Dev")

if start_button:
    video_stream.start_video_feed()

if stop_button:
    video_stream.stop_video_feed()
