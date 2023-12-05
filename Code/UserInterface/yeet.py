import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import psutil  # For CPU and memory usage
from pathlib import Path
from typing import Dict
import queue
import numpy as np
from MethodizedEulerian import EulerianMagnification

HERE = Path(__file__).parent


class VideoProcessor(VideoProcessorBase):
    result_queue: "queue.Queue[Dict[str, float]]"

    def __init__(self, method=None, brightness=None, contrast=None) -> None:
        self.result_queue = queue.Queue()
        self.method = method
        self.brightness = brightness
        self.contrast = contrast
        self.magnification_processor = EulerianMagnification()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        # Apply brightness, color changes, and contrast adjustments
        adjusted_frame = cv2.convertScaleAbs(
            img, alpha=self.contrast, beta=self.brightness
        )

        if self.method == "Deep Learning Frameworks":
            # Convert to grayscale
            gray = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2GRAY)

            # Convert back to BGR for displaying
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # Ensure the correct shape for av.VideoFrame
            gray_bgr = np.ascontiguousarray(gray_bgr)

            # Create av.VideoFrame with the correct format
            new_frame = av.VideoFrame.from_ndarray(gray_bgr, format="bgr24")
        elif self.method == "Traditional Eulerian Magnification":
            # Perform Eulerian magnification using the class and display the output.
            processed_frame = self.magnification_processor.process_frame(adjusted_frame)
            new_frame = av.VideoFrame.from_ndarray(processed_frame, format="bgr24")
        else:
            new_frame = av.VideoFrame.from_ndarray(adjusted_frame, format="bgr24")

        # Perform CPU and memory usage calculations
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        result = {"CPU Usage": cpu_percent, "Memory Usage": memory_percent}

        # NOTE: This `recv` method is called in another thread,
        # so it must be thread-safe.
        self.result_queue.put(result)

        return new_frame


def app_system_monitor():
    st.sidebar.header("WebRTC Configuration")

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

    # Create a class instance for the Eulerian Magnification
    METHOD = st.sidebar.selectbox(
        "Which algorithm do you want to use?",
        (
            "None",
            "Traditional Eulerian Magnification",
            "Custom implemented Eulerian Magnification",
            "Deep Learning Frameworks",
        ),
    )

    webrtc_ctx = webrtc_streamer(
        key="system-monitor",
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )

    st.sidebar.header("System Monitor Configuration")
    if st.sidebar.checkbox("Show CPU and Memory Usage", value=True):
        if webrtc_ctx.state.playing:
            monitor_placeholder = st.empty()
            # NOTE: The video transformation with system monitoring and
            # this loop displaying the result values are running
            # in different threads asynchronously.
            # Then the rendered video frames and the values displayed here
            # are not strictly synchronized.
            while True:
                if webrtc_ctx.video_processor:
                    try:
                        result = webrtc_ctx.video_processor.result_queue.get(
                            timeout=1.0
                        )
                        # Update Sliders
                        webrtc_ctx.video_processor.brightness = brightness
                        webrtc_ctx.video_processor.contrast = contrast
                        webrtc_ctx.video_processor.method = METHOD

                    except queue.Empty:
                        result = None
                    monitor_placeholder.text(
                        f"CPU Usage: {result['CPU Usage']:.2f}% | Memory Usage: {result['Memory Usage']:.2f}%"
                    )
                else:
                    break


if __name__ == "__main__":
    app_system_monitor()
