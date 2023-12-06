import streamlit as st
from webcam_use import app_system_monitor
from demo import demo
from Video_upload import video_upload


# Helper method for importing Markdown files
def markdownreader(file):
    with open("Text/" + file) as f:
        lines = f.readlines()
        for line in lines:
            st.markdown(line, unsafe_allow_html=True)


# Hide first Radio Button
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


### Main functionality
with st.sidebar:
    st.title("Heart rate estimation in video data")
    st.text("Select type")
    implementation = st.radio(
        "Choose algorithm framework: ",
        [" ", "Pre-recorded video", "Live-Feed", "Demo", "Main"],
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
                "MTTS",
                "HR_CNN"
            ),
        )
        st.text("Beware, MTTS and HR_CNN computes the \n HR beforehand and is therfore \n computationally heavy with long videos.")
        st.text("If video is too long, memory errors \n may occur.")
    # if implementation == "Demo":
    #   demo()
if implementation == " " or implementation == "Main":
    markdownreader("Main.md")

if implementation == "Pre-recorded video":
    video_upload(video_data=video_file, method=method)
if implementation == "Live-Feed":
    app_system_monitor()
if implementation == "Demo":
    demo()
