import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings
import threading
import cv2
import numpy as np
from PIL import Image
import av
from MethodizedEulerian import EulerianMagnification


def webcam_use():
    magnification_processor = EulerianMagnification()
    st.header("Webcam live feed")
    st.text("Select Method")
    METHOD = st.sidebar.selectbox(
        "Which algorithm do you want to use?",
        (
            "None",
            "Traditional Eulerian Magnification",
            "Custom implemented Eulerian Magnification",
            "Deep Learning Frameworks",
        ),
    )

    def callback(frame):
        img = frame.to_ndarray(format="bgr24")
        if METHOD == "Deep Learning Frameworks":
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Convert back to BGR for displaying
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # Ensure the correct shape for av.VideoFrame
            gray_bgr = np.ascontiguousarray(gray_bgr)

            # Create av.VideoFrame with the correct format
            new_frame = av.VideoFrame.from_ndarray(gray_bgr, format="bgr24")
        elif METHOD == "Traditional Eulerian Magnification":
            # Perform eulerian magnification using the class and display the output.

            processed_frame = magnification_processor.process_frame(img)
            new_frame = av.VideoFrame.from_ndarray(processed_frame, format="bgr24")
        else:
            new_frame = av.VideoFrame.from_ndarray(img, format="bgr24")

        return new_frame

    webrtc_streamer(key="example", video_frame_callback=callback)
