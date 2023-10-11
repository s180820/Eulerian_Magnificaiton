import streamlit as st
from facerecog import *
import tempfile

st.title("Hello World")
st.write("This is my first Streamlit app!")


f = st.file_uploader('Upload a video file', type=['mp4'])

if f is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(f.read())
    st.video(f)
    network = cv2.dnn.readNetFromCaffe("models/deploy.prototxt.txt", "models/res10_300x300_ssd_iter_140000.caffemodel")
    st.text("Creating video...")
    facevid = start_video_feed(tfile.name, network, save_video=True)
    st.video("filename.avi")


