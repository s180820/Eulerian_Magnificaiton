import streamlit as st
import pandas as pd
import numpy as np
import cv2

st.set_page_config(page_title="Eulerian Magnification", page_icon=":eyeglasses:")
tab1, tab2, tab3 = st.tabs(["Pre-recorded video", "Live feed", "About"])


## Helper functions ""
def markdownreader(file):
    with open("Text/" + file) as f:
        lines = f.readlines()
        for line in lines:
            st.markdown(line, unsafe_allow_html=True)


## TABS ##
with tab1:
    markdownreader("Main.md")
    # Data uploade

    data = st.file_uploader(" ", accept_multiple_files=False, type=["mp4"])
    if not data:
        st.write("Upload a file before continuing")
    else:
        st.video(data)


with tab2:
    # Init
    markdownreader("Webcam.md")
    start_button_pressed = st.button("Start")

    stop_bottom_pressed = st.button("Stop")

    frame_placeholder = st.empty()

    if start_button_pressed:
        cap = cv2.VideoCapture(0)
        while cap.isOpened() and not stop_bottom_pressed:
            ret, frame = cap.read()
            if not ret:
                st.write("The video capture has ended")
                break
            frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )  # RGB Format to support streamlit
            frame_placeholder.image(frame, channels="RGB")
        cap.release()
        cv2.destroyAllWindows()


with tab3:
    markdownreader("Background.md")
