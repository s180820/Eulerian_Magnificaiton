import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import plotly.express as px  # interactive charts
import tempfile
from MethodizedEulerian import EulerianMagnification
from multiface import MultifaceEulerianMagnification

def video_upload(video_data=None, method="Traditional Eulerian Magnification"):
    st.title("Video upload")
    placeholder = st.empty()
    if method == "Traditional Eulerian Magnification":
        eulerian_processor = EulerianMagnification()
    elif method == "Custom implemented Eulerian Magnification":
        eulerian_processor = MultifaceEulerianMagnification()
    else:
        print("No method selected")
        st.write("No method selected")
        return
    bpms=[]
    bpms_second = [0]

    time_idx = 1

    if not video_data:
        st.write("Upload a file before continuing")
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_data.read())
        cap = cv2.VideoCapture(tfile.name)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("The video capture has ended")
                break
            #img = frame.to_ndarray(format="bgr24")
            if method == "Traditional Eulerian Magnification":
                eulerian_processor.process_frame(frame)
            elif method == "Custom implemented Eulerian Magnification":
                eulerian_processor.process_frame_streamlit(frame)
            bpmES = eulerian_processor.get_bpm_over_time()
            
            

            bpms.append(bpmES.mean())
                #if len(bpms) > 200:
                    #   bpms = bpms[-200:]
                # get hr for each second
            if len(bpms) > 30:
                bpm = np.mean(bpms[-30:])
                bpms_second.append(bpm)
                #new_row = pd.DataFrame([[time_idx, bpm, "Eulerian"]], columns=["Time", "BPM", "type"])
                #plot_df = pd.concat([plot_df, new_row])
                time_idx += 1
                bpms = []
            
            with placeholder.container():
                fig_col1, fig_col2 = st.columns(2)
                with fig_col1:
                    st.markdown("### BPM over time")
                    st.image(frame, channels="RGB")
                with fig_col2:
                    st.markdown("### BPM Statistics")
                    bpm_df = pd.DataFrame(bpms_second, columns=["BPM_Eulerian"])
                    bpm_df = bpm_df.describe()
                    bpm_df = bpm_df.drop(["count", "min", "25%", "50%", "75%"])
                    bpm_df = bpm_df.rename(index={"mean": "Mean", "max": "Max", "std": "Std"})
                    bpm_df = bpm_df.T
                    bpm_df = bpm_df.round(2)
                    st.write(bpm_df)

                fig = px.line(x=np.arange(len(bpms_second)), y=bpms_second, labels={"x": "Time", "y": "BPM"})
                #fig = px.line(plot_df[plot_df["Time"].astype(int) > 6], x="Time", y="BPM", 
                 #                       color="type", labels={"x": "Time", "y": "BPM"})
                st.write(fig)
