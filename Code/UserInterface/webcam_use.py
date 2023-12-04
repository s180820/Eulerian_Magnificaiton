import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import numpy as np
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

    # Sliders for brightness and contrast
    brightness = st.sidebar.slider(
        "Brightness",
        min_value=-100,
        max_value=100,
        value=0,
    )
    contrast = st.sidebar.slider(
        "Contrast",
        min_value=0.1,
        max_value=3.0,
        value=1.0,
        step=0.1,
    )

    def callback(frame):
        img = frame.to_ndarray(format="bgr24")
        # Apply brightness, color changes, and contrast adjustments
        adjusted_frame = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
        img = adjusted_frame
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
            processed_frame = magnification_processor.process_frame(adjusted_frame)
            new_frame = av.VideoFrame.from_ndarray(processed_frame, format="bgr24")
        else:
            new_frame = av.VideoFrame.from_ndarray(adjusted_frame, format="bgr24")

        return new_frame

    webrtc_streamer(key="example", video_frame_callback=callback)
