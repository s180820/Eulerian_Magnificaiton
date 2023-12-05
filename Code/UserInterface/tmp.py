import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import psutil
from pathlib import Path
import queue
import numpy as np
from MethodizedEulerian import EulerianMagnification
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px  # interactive charts
from multiface import MultifaceEulerianMagnification

# Import your MultifaceEulerianMagnification class from the provided code


class VideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        super().__init__()
        self.magnification_processor = MultifaceEulerianMagnification()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        # Apply brightness, color changes, and contrast adjustments
        adjusted_frame = cv2.convertScaleAbs(
            img, alpha=self.contrast, beta=self.brightness
        )
        if self.method == "Traditional Eulerian Magnification":
            # Perform Eulerian magnification using the class and display the output.
            print("Hi there")
            pass
        elif self.method == "Custom implemented Eulerian Magnification":
            processed_frame2 = self.magnification_processor.process_frame_streamlit(
                adjusted_frame
            )
            print("[Debugging]", processed_frame2)
            # new_frame = av.VideoFrame.from_ndarray(processed_frame2, format="bgr24")


def app_system_monitor():
    st.sidebar.header("WebRTC Configuration")

    # Create a class instance for the Eulerian Magnification
    METHOD = st.sidebar.selectbox(
        "Which algorithm do you want to use?",
        (
            "None",
            "Traditional Eulerian Magnification",
            "Custom implemented Eulerian Magnification",
        ),
    )
    webrtc_ctx = webrtc_streamer(
        key="system-monitor",
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )


if __name__ == "__main__":
    app_system_monitor()
