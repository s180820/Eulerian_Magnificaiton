import streamlit as st
from streamlithelper import StreamlitHelper
from webcam_use import app_system_monitor
from demo import demo
from Video_upload import video_upload

shelper = StreamlitHelper()

st.markdown(
    """
    <style>
        div[role=radiogroup] label:first-of-type {
            visibility: hidden;
            height: 0px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Heart rate estimation in video data")
### Main functionality
with st.sidebar:
    st.title("Heart rate estimation in video data")
    st.text("Select type")
    implementation = st.radio(
        "Choose algorithm framework: ", [" ", "Pre-recorded video", "Live-Feed", "Demo"]
    )
    if implementation == "Pre-recorded video":
        st.text("Select video")
        video_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])

        if video_file is not None:
            st.video(video_file)
        st.text("Select Method")
        method = st.selectbox(
            "Which algorithm do you want to use?",
            (
                "None",
                "Traditional Eulerian Magnification",
                "Custom implemented Eulerian Magnification",
            ),
        )
    # if implementation == "Demo":
    #   demo()


if implementation == "Pre-recorded video":
    video_upload(video_data=video_file, method=method)
if implementation == "Live-Feed":
    app_system_monitor()
if implementation == "Demo":
    demo()
