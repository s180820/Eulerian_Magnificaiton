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

HERE = Path(__file__).parent


class VideoProcessor(VideoProcessorBase):
    result_queue: "queue.Queue[Dict[str, float]]"
    cpu_usage_history: list

    def __init__(
        self, method=None, brightness=None, contrast=None, display_pyramid=None
    ) -> None:
        self.result_queue = queue.Queue()
        self.method = method
        self.brightness = brightness
        self.contrast = contrast
        self.display_pyramid = display_pyramid
        self.magnification_processor = EulerianMagnification()
        self.custom_magnification_processor = MultifaceEulerianMagnification()
        self.cpu_usage_history = []
        self.memory_usage_history = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        # Apply brightness, color changes, and contrast adjustments
        adjusted_frame = cv2.convertScaleAbs(
            img, alpha=self.contrast, beta=self.brightness
        )
        if self.method == "Traditional Eulerian Magnification":
            # Perform Eulerian magnification using the class and display the output.
            processed_frame = self.magnification_processor.process_frame(
                adjusted_frame, self.display_pyramid
            )
            new_frame = av.VideoFrame.from_ndarray(processed_frame, format="bgr24")
        elif self.method == "Custom implemented Eulerian Magnification":
            processed_frame2 = (
                self.custom_magnification_processor.process_frame_streamlit(
                    adjusted_frame
                )
            )
            new_frame = av.VideoFrame.from_ndarray(processed_frame2, format="bgr24")

        else:
            new_frame = av.VideoFrame.from_ndarray(adjusted_frame, format="bgr24")

        # Perform CPU and memory usage calculations
        cpu_percent = psutil.cpu_percent()

        memory_percent = psutil.virtual_memory().percent

        result = {"CPU Usage": cpu_percent, "Memory Usage": memory_percent}
        # Append CPU, Memory usage to history
        self.cpu_usage_history.append(cpu_percent)
        self.memory_usage_history.append(memory_percent)

        if len(self.cpu_usage_history) or len(self.memory_usage_history) > 150:
            self.cpu_usage_history = self.cpu_usage_history[-150:]
            self.memory_usage_history = self.memory_usage_history[-150:]

        # NOTE: This `recv` method is called in another thread,
        # so it must be thread-safe.
        self.result_queue.put(result)

        return new_frame

    def getBPM(self):
        return self.magnification_processor.get_bpm_over_time()

    def getBPMCustom(self):
        return self.custom_magnification_processor.get_bpm_over_time()


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

    # Checkbox for displaying pyramid in Traditional Eulerian Magnification
    display_pyramid = st.sidebar.checkbox(
        "Display Pyramid (Traditional Eulerian Magnification)",
        value=True,
        key="display_pyramid",
    )

    webrtc_ctx = webrtc_streamer(
        key="system-monitor",
        video_processor_factory=VideoProcessor,
        async_processing=True,
        media_stream_constraints={"video": True, "audio": False},
    )

    st.sidebar.header("System Monitor Configuration")
    checkbox = st.sidebar.checkbox("Show CPU and Memory Usage", value=True)
    if webrtc_ctx.state.playing:
        dataframe_placeholder = st.empty()
        dataframe_placeholder_custom = st.empty()
        monitor_placeholder = st.empty()
        system_placeholder = st.empty()
        # NOTE: The video transformation with system monitoring and
        # this loop displaying the result values are running
        # in different threads asynchronously.
        # Then the rendered video frames and the values displayed here
        # are not strictly synchronized.

        frame_counter = 0
        while True:
            if webrtc_ctx.video_processor:
                try:
                    result = webrtc_ctx.video_processor.result_queue.get(timeout=1.0)
                    # Update Sliders
                    webrtc_ctx.video_processor.brightness = brightness
                    webrtc_ctx.video_processor.contrast = contrast
                    webrtc_ctx.video_processor.method = METHOD
                    webrtc_ctx.video_processor.display_pyramid = display_pyramid

                    frame_counter += 1
                    if METHOD == "Traditional Eulerian Magnification":
                        bpm_values = webrtc_ctx.video_processor.getBPM()
                        if len(bpm_values) > 200:
                            bpm_values = bpm_values[-200:]
                        bpm_df = pd.DataFrame(bpm_values, columns=["BPM"])
                        bpm_df = bpm_df.describe().drop(
                            ["count", "min", "25%", "50%", "75%"]
                        )
                        bpm_df.rename(
                            index={
                                "mean": "Mean",
                                "max": "Max",
                                "std": "Standard Deviation",
                            },
                            inplace=True,
                        )
                        bpm_df = bpm_df.T.round(2)
                    if METHOD == "Custom implemented Eulerian Magnification":
                        bpm_values_custom = webrtc_ctx.video_processor.getBPMCustom()
                        # Remove the first element (where the key is None)
                        bpm_values_custom = {
                            key: value
                            for key, value in bpm_values_custom.items()
                            if key is not None
                        }
                        person_stats = {}

                        for person, value in bpm_values_custom.items():
                            person_stats[person] = {
                                "Mean": pd.Series(value).mean(),
                                "Max": pd.Series(value).max(),
                                "Std": pd.Series(value).std(),
                            }
                            # Create a dataframe from the dictionary
                        df_custom = pd.DataFrame.from_dict(person_stats, orient="index")
                        df_custom.index.name = "Person"  # Name index col

                except queue.Empty:
                    result = None
                if METHOD == "Traditional Eulerian Magnification":
                    with dataframe_placeholder.container():
                        fig_col1, fig_col2 = st.columns(2)
                        with fig_col1:
                            st.markdown("## BPM statistics")
                            st.write(bpm_df)
                        with fig_col2:
                            fig = px.line(
                                x=np.arange(len(bpm_values)),
                                y=bpm_values,
                                labels={"X": "Frame", "y": "BPM"},
                            )
                            st.write(fig)
                if METHOD == "Custom implemented Eulerian Magnification":
                    with dataframe_placeholder_custom.container():
                        fig_col1, fig_col2 = st.columns(2)
                        with fig_col1:
                            st.markdown("## BPM statistics")
                            st.write(df_custom.round(2))
                if checkbox:
                    with system_placeholder.container():
                        st.markdown("## System stats")
                        st.write(
                            f"CPU Usage: {result['CPU Usage']:.2f}% | Memory Usage: {result['Memory Usage']:.2f}%"
                        )

            else:
                break


if __name__ == "__main__":
    app_system_monitor()
