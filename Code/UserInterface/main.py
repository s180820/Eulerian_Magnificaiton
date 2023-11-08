import streamlit as st
import pandas as pd
import numpy as np
import cv2

st.set_page_config(page_title="Eulerian Magnification", page_icon=":eyeglasses:")
tab1, tab2 = st.tabs(["Main", "Background"])


def markdownreader(file):
    with open("Text/" + file) as f:
        lines = f.readlines()
        for line in lines:
            st.markdown(line, unsafe_allow_html=True)


with tab1:
    markdownreader("Main.md")
    # Data uploade

    data = st.file_uploader(" ", accept_multiple_files=False, type=["mp4"])
    if not data:
        st.write("Upload a file before continuing")
    else:
        st.video(data)


with tab2:
    markdownreader("Background.md")
