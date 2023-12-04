from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings
import streamlit as st
import threading
import cv2
from MethodizedEulerian import EulerianMagnification


class LiveFeedWebcam(VideoTransformerBase):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        st.header("Webcam Live feed")
        self.model_lock = threading.Lock()
